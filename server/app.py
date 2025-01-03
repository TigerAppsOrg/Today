from flask import Flask, request, jsonify, send_from_directory, \
    session, redirect, url_for, abort, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from mixpanel import Mixpanel
from clients import db_client
from datetime import datetime, timezone
from models import Event, ChatQueryInput, Feedback
from tools import tools, choose_tool_and_rewrite, invoke_tool
from response import generate_response
from memory import Memory, ToolInvocation, MessageType
from cas import CASClient
from urllib.parse import urlparse, parse_qs
import os
import uuid
import time

load_dotenv()

app = Flask(__name__, static_folder='dist', static_url_path='/')
app.secret_key = os.getenv("APP_SECRET_KEY")

mp = Mixpanel(os.getenv("MIXPANEL"))

CORS(app, resources={r"/*": {"origins": "*"}})

cas_client = CASClient()

def authenticated():
    is_logged_in = cas_client.is_logged_in()
    if is_logged_in:
        netid = cas_client.authenticate()
        db_client["logins"].insert_one({
            "netid": netid, 
            "time": int(time.time())
        })

    return is_logged_in

# ========== UI ==========

@app.route('/static/<path:filename>')
def serve_static(filename):
    if not authenticated():
        abort(401)
    return send_from_directory('dist', filename)

@app.route('/')
def index():
    if not authenticated():
        return redirect(url_for("login"))
    return send_from_directory('dist', 'index.html')

@app.route("/login")
def login():
    cas_client.authenticate()
    next = url_for("index")
    if "next" in session:
        next = session.pop("next")
    return redirect(next)

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

# ========== EXTENSION ==========

@app.route('/api/extension/widget-data', methods=['GET'])
def widget_data():
    data = db_client['widgets'].find_one({'_id': 'data'})
    if data:
        data["timestamp"] = str(datetime.now(timezone.utc))
    return jsonify(data)

@app.route('/api/track', methods=['POST'])
def track():
    data = request.get_json()
    data = Event(**data)
    mp.track(data.uuid, data.event, data.properties)
    return '', 204

# ========== CHATBOT ==========

@app.route('/api/feedback', methods=['POST'])
def feedback():
    if not authenticated():
        abort(401)
    data = request.get_json()
    feedback = Feedback(**data)
    collection = db_client["feedback"]

    if not feedback.feedback:
        collection.delete_one({
            "uuid": feedback.uuid,
            "session_id": feedback.session_id,
            "msg_index": feedback.msg_index
        })
    else:
        filter = {
            "uuid": feedback.uuid,
            "session_id": feedback.session_id,
            "msg_index": feedback.msg_index
        }
        new_document = {
            "uuid": feedback.uuid,
            "session_id": feedback.session_id,
            "msg_index": feedback.msg_index,
            "feedback": feedback.feedback,
            "time": int(time.time())
        }

        collection.replace_one(filter, new_document, upsert=True)

    return '', 200
    
@app.route('/api/chat', methods=['POST'])
def chat():
    if not authenticated():
        abort(401)
    data = request.get_json()
    query = ChatQueryInput(**data)
    memory = Memory(query.uuid, query.session_id)

    tool, query_rewrite = choose_tool_and_rewrite(tools, memory, query.text)
    tool_result = invoke_tool(tool, query_rewrite)
    tool_use = ToolInvocation(
        tool=tool,
        input=query_rewrite,
        output=tool_result
    )

    memory.add_message(MessageType.HUMAN, query.text)
    mp.track(query.uuid, "chat", {'session_id': query.session_id})

    return generate_response(memory, tool_use), {"Content-Type": "text/plain"}

if __name__ == '__main__':
    app.run(host="localhost", port=6001, debug=True)
