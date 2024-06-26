import { useState } from "react";
import React from "react";
import { useStorage } from "../context/StorageContext";

function Name() {
  const storage = useStorage();

  const [text, setText] = useState(storage.getLocalStorageDefault("name", ""));
  const [inputValue, setInputValue] = useState(
    storage.getLocalStorageDefault("name", "")
  );
  const [showPopup, setShowPopup] = useState(text === "");

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const name = inputValue;
    if (name !== "") {
      storage.setLocalStorage("name", name);
      setText(name);
      setShowPopup(false);
    }
  };

  return (
    <span>
      <span className="name" onClick={() => setShowPopup(true)}>
        {text}
      </span>
      {showPopup && (
        <div className="popup">
          <form onSubmit={handleSubmit}>
            <label>Please enter your name:&nbsp;&nbsp;</label>
            <input
              type="text"
              value={inputValue}
              onChange={handleInputChange}
            />
            <button type="submit">Submit</button>
          </form>
        </div>
      )}
    </span>
  );
}

export default Name;
