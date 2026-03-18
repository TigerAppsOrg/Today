import React, { useState } from "react";
import Tay from "../images/tay.png";
import SlidingPane from "react-sliding-pane";
import { StorageKeys, useStorage } from "../context/StorageContext";
import { Form, Button } from "react-bootstrap";
import { FiSend } from "react-icons/fi";
import config from "../config";
import { EventTypes, useMixpanel } from "../context/MixpanelContext";
import { useTime } from "../context/TimeContext";

function Chat() {
  const storage = useStorage();
  const mixpanel = useMixpanel();
  const time = useTime();

  const [isPaneOpen, setPaneOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [query, setQuery] = useState('');

  const handleSubmit = (e: any) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    setQuery(inputValue);
    setInputValue('');
    setPaneOpen(true);
    mixpanel.trackEvent(EventTypes.OPENED_CHAT, time.getUTC().toString())
  };

  return (
    <div style={{ padding: "24px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
        <img
          id="tay-img"
          src={Tay}
          style={{ width: "80px", height: "80px", borderRadius: "50%", cursor: "pointer", objectFit: "cover" }}
          onClick={() => setPaneOpen(true)}
          alt="Tay Chatbot"
        />
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: "16px", fontWeight: "600", marginBottom: "10px", color: "#f5f5f7" }}>
            Hey! My name is Tay, and I'm an AI assistant.
          </div>
          <Form onSubmit={handleSubmit} style={{ display: "flex", width: "100%" }}>
            <Form.Control
              type="text"
              id="chat-input"
              placeholder="Ask me anything..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              style={{ 
                flex: 1, 
                borderTopRightRadius: 0, 
                borderBottomRightRadius: 0,
                backgroundColor: "rgba(255, 255, 255, 0.1)",
                border: "1px solid rgba(255, 255, 255, 0.2)",
                color: "white"
              }}
            />
            <Button
              variant="primary"
              id="chat-button"
              type="submit"
              style={{ 
                borderTopLeftRadius: 0, 
                borderBottomLeftRadius: 0,
                backgroundColor: "#E77500",
                borderColor: "#E77500"
              }}
            >
              <FiSend />
            </Button>
          </Form>
        </div>
      </div>

      <SlidingPane
        isOpen={isPaneOpen}
        onRequestClose={() => {
          setPaneOpen(false);
        }}
        width="640px"
      >
        <iframe
          src={config.URL + `?uuid=${storage.getLocalStorage(
            StorageKeys.UUID
          )}&query=${query}`}
          width="100%"
          height="100%"
          style={{ border: "none" }}
          title="Chatbot"
        />
      </SlidingPane>
    </div>
  );
}

export default Chat;
