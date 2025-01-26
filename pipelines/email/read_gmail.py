from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import base64
import os
from preprocess import get_expiry_time, is_eating_club
import time
import hashlib
from protonmail import ProtonMail
from protonmail.models import Message
import re

load_dotenv()

username = os.getenv("PROTON_USERNAME")
password = os.getenv("PROTON_PASSWORD")

def extract_text_and_links_from_html(html_string: str):
    soup = BeautifulSoup(html_string, 'html.parser')
    text = soup.get_text(strip=True)
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return text, list(set(links))

def compute_email_hash(subject, sender, body):
    email_content = f"{subject}{sender}{body}"
    return hashlib.md5(email_content.encode('utf-8')).hexdigest()

def read_email(message: Message):
    text_parts = []
    text, extracted_links = extract_text_and_links_from_html(message.body)
    ids = []
    docs = []

    # extract headers case-sensitively
    subject = message.subject.strip().lower()
    sender = message.sender.address
    email_timestamp = message.time

    # Emphasize the subject line
    text_parts.append(f"SUBJECT: {subject}")
    text_parts.append(f"From: {sender}")
    text_parts.append(f"Body: {text}")

    page_content = "\n".join(text_parts)

    # Compute 'expiry_time' using 'get_expiry_time'
    expiry_time = get_expiry_time(page_content, email_timestamp)

    # Determine if the email is from an eating club
    eating_club = is_eating_club(page_content)

    # Use message_id as the document ID
    doc_id = message.id

    # 'received_time' field to store when the email was processed
    # Will help prioritize recent emails
    received_time = int(time.time())

    # Include 'subject' in the metadata
    metadata = {
        "subject": subject,
        "links": extracted_links,
        "time": email_timestamp,
        "expiry": expiry_time,
        "source": "email",
        "received_time": received_time
    }

    docs.append(Document(
        page_content=page_content,
        metadata=metadata
    ))
    ids.append(doc_id)

    if eating_club:
        # Modify the 'source' for eating club emails
        metadata_ec = metadata.copy()
        metadata_ec["source"] = "eatingclub"
        docs.append(Document(
            page_content=page_content,
            metadata=metadata_ec
        ))
        ids.append(doc_id + "__ec")

    return {
        "ids": ids,
        "docs": docs,
        "subject": subject
    }

def main():
    is_dry_run = False
    
    # list of emails/listservs to check
    # email_addresses = [
    #     "WHITMANWIRE@princeton.edu",
    #     "westwire@princeton.edu",
    #     "allug@princeton.edu",
    #     "freefood@princeton.edu",
    #     "matheymail@princeton.edu",
    #     "public-lectures@princeton.edu",
    #     "CampusRecInfoList@princeton.edu",
    #     "pace-center@princeton.edu",
    #     "tigeralert@princeton.edu",
    # ]
    proton = ProtonMail()
    proton.login(username, password)

    all_messages: list[Message] = []
    messages = proton.get_messages()

    for message in messages:
        msg = proton.read_message(message)
        if msg.unread == 0:
            break
        if msg.recipients[0].address == "WHITMANWIRE@Princeton.EDU":
            all_messages.append(msg)

    print(f"Total messages fetched: {len(all_messages)}")

    if not all_messages:
        print("[INFO] No unread emails found.")
        return

    processed_messages = []
    processed_subjects = set()
    ids = []
    docs = []

    for message in all_messages:
        # normalize subject header (sometimes it's caps and sometimes not)
        # i genuinely have no idea why
        subject = message.subject.strip().lower()

        # normalize subject line
        normalized_subject = ' '.join(subject.split())

        # skip if processed
        if normalized_subject in processed_subjects:
            print(f"[INFO] Skipping duplicate email with subject: {subject}")
            continue
        else:
            processed_subjects.add(normalized_subject)
            print(f"[INFO] Processing email with subject: {subject}")

        msg_data = read_email(message)
        if not msg_data:
            continue  # skip if email content couldnt be read

        processed_messages.append(msg_data)

        ids.extend(msg_data["ids"])
        docs.extend(msg_data["docs"])

    print(f"Total unique messages to add: {len(docs)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=256)

    client = MongoClient(os.getenv("MONGO_CONN"))
    # Define collection and index name
    db_name = "today"
    collection_name = "crawl"
    atlas_collection = client[db_name][collection_name]

    vector_store = MongoDBAtlasVectorSearch(
        atlas_collection,
        embeddings
    )

    if not is_dry_run:
        try:
            if docs:
                # before adding documents, check for existing IDs
                existing_ids = set()
                existing_docs_cursor = atlas_collection.find({'_id': {'$in': ids}}, {'_id': 1})
                for doc in existing_docs_cursor:
                    existing_ids.add(doc['_id'])

                # filter out documents with existing ids
                new_docs = []
                new_ids = []
                for doc, id_ in zip(docs, ids):
                    if id_ not in existing_ids:
                        new_docs.append(doc)
                        new_ids.append(id_)
                    else:
                        print(f"[INFO] Skipping document with duplicate _id: {id_}")

                if new_docs:
                    vector_store.add_documents(new_docs, ids=new_ids)
                    print(f"[INFO] Added {len(new_docs)} new email documents:")
                    for doc in new_docs:
                        print(f" - {doc.metadata.get('subject', 'No Subject')}")
                else:
                    print("[INFO] All documents already exist in the collection.")
            else:
                print("[INFO] No new documents to add.")
        except Exception as e:
            print(f"[ERROR] Failed to process emails: {e}")
    else:
        print("[INFO] Finished email dry run")

if __name__ == '__main__':
    main()
