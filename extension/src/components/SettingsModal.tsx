import React, { useEffect, useState } from "react";
import { StorageKeys, useStorage } from "../context/StorageContext";
import "./SettingsModal.css";

type Props = {
  isOpen: boolean;
  onClose: () => void;
  activeWidgets: string[];
  toggleWidget: (id: string) => void;
  availableWidgets: { id: string; name: string }[];
  columnCount: 2 | 3;
  onColumnCountChange: (count: 2 | 3) => void;
};

const SettingsModal: React.FC<Props> = ({
  isOpen,
  onClose,
  activeWidgets,
  toggleWidget,
  availableWidgets,
  columnCount,
  onColumnCountChange,
}) => {
  const storage = useStorage();
  const [activeTab, setActiveTab] = useState<"widgets" | "background">("widgets");
  const [customBackground, setCustomBackground] = useState(() => {
    return storage.getLocalStorage(StorageKeys.CUSTOM_BG) || "";
  });
  const [isRendered, setIsRendered] = useState(isOpen);
  const [isClosing, setIsClosing] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setIsRendered(true);
      setIsClosing(false);
      return;
    }

    if (isRendered) {
      setIsClosing(true);
      const timeout = window.setTimeout(() => {
        setIsRendered(false);
        setIsClosing(false);
      }, 220);
      return () => window.clearTimeout(timeout);
    }
  }, [isOpen, isRendered]);

  useEffect(() => {
    if (!isOpen) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isRendered) return null;

  const applyBackground = (url: string) => {
    if (url) {
      document.body.style.backgroundImage = `url('${url}')`;
      document.body.style.backgroundSize = "cover";
      document.body.style.backgroundPosition = "center center";
      document.body.style.backgroundAttachment = "fixed";
      return;
    }
    document.body.style.backgroundImage = "";
    document.body.style.backgroundSize = "";
    document.body.style.backgroundPosition = "";
    document.body.style.backgroundAttachment = "";
  };

  const handleCustomBackground = (url: string) => {
    storage.setLocalStorage(StorageKeys.CUSTOM_BG, url);
    setCustomBackground(url);
    applyBackground(url);
  };

  const handleBackgroundUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      if (!result) return;
      handleCustomBackground(result);
    };
    reader.readAsDataURL(file);
  };

  const handleClearBg = () => {
    window.localStorage.removeItem(StorageKeys.CUSTOM_BG);
    setCustomBackground("");
    applyBackground("");
  };

  return (
    <div className={`settings-modal-overlay ${isClosing ? "closing" : ""}`} onClick={onClose}>
      <div
        className={`settings-modal ${isClosing ? "closing" : ""}`}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Customization settings"
      >
        <button className="settings-modal-close" onClick={onClose} aria-label="Close settings">
          ×
        </button>

        <div className="settings-modal-head">
          <h4 className="settings-modal-title">Customize</h4>
          <div className="settings-modal-segmented" role="tablist" aria-label="Settings sections">
            <button
              className={`settings-modal-segment ${activeTab === "widgets" ? "active" : ""}`}
              onClick={() => setActiveTab("widgets")}
              role="tab"
              aria-selected={activeTab === "widgets"}
            >
              Widgets
            </button>
            <button
              className={`settings-modal-segment ${activeTab === "background" ? "active" : ""}`}
              onClick={() => setActiveTab("background")}
              role="tab"
              aria-selected={activeTab === "background"}
            >
              Background
            </button>
          </div>
        </div>

        <div className="settings-modal-content">
          {activeTab === "widgets" && (
            <section className="settings-panel" aria-label="Manage widgets">
              <p className="settings-panel-subtitle">Choose what appears on your dashboard.</p>
              <div className="settings-widget-group">
                {availableWidgets.map((widget) => {
                  const checked = activeWidgets.includes(widget.id);
                  return (
                    <label key={widget.id} className="settings-widget-row">
                      <span className="settings-widget-name">{widget.name}</span>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleWidget(widget.id)}
                        className="settings-widget-toggle"
                      />
                    </label>
                  );
                })}
              </div>

              <p className="settings-panel-subtitle">Pick your desktop column layout.</p>
              <div className="settings-layout-toggle" role="radiogroup" aria-label="Widget columns">
                <button
                  type="button"
                  className={`settings-layout-option ${columnCount === 2 ? "active" : ""}`}
                  onClick={() => onColumnCountChange(2)}
                  role="radio"
                  aria-checked={columnCount === 2}
                >
                  2 columns
                </button>
                <button
                  type="button"
                  className={`settings-layout-option ${columnCount === 3 ? "active" : ""}`}
                  onClick={() => onColumnCountChange(3)}
                  role="radio"
                  aria-checked={columnCount === 3}
                >
                  3 columns (default)
                </button>
              </div>
            </section>
          )}

          {activeTab === "background" && (
            <section className="settings-panel" aria-label="Background options">
              <p className="settings-panel-subtitle">Upload your own image for the background.</p>
              <div className="settings-upload-card">
                <label className="settings-upload-label" htmlFor="custom-bg-upload">
                  Choose Image
                </label>
                <input
                  id="custom-bg-upload"
                  type="file"
                  accept="image/*"
                  className="settings-upload-input"
                  onChange={handleBackgroundUpload}
                />
                {customBackground && <span className="settings-upload-hint">Custom image applied.</span>}
              </div>

              <button
                type="button"
                className="settings-action-button settings-action-secondary"
                onClick={handleClearBg}
              >
                Use Default
              </button>
            </section>
          )}
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
