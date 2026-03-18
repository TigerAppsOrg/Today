import React, { useEffect, useMemo, useState } from "react";
import "./TigerLinks.css";
import { StorageKeys, useStorage } from "../context/StorageContext";
import { FiTrash2 } from "react-icons/fi";

type LinkItem = {
  id: string;
  name: string;
  url: string;
  iconUrl?: string;
  removable?: boolean;
};

type SavedLink = {
  id: string;
  name: string;
  url: string;
};

const TIGER_LINKS: LinkItem[] = [
  {
    id: "canvas",
    name: "Canvas",
    url: "https://princeton.instructure.com/",
    iconUrl: "https://www.instructure.com/sites/default/files/image/2025-07/canvas_icon_color_rgb.png",
  },
  {
    id: "gradescope",
    name: "Gradescope",
    url: "https://www.gradescope.com/",
    iconUrl: "https://www.google.com/s2/favicons?sz=64&domain=gradescope.com",
  },
  {
    id: "gmail",
    name: "Gmail",
    url: "https://mail.google.com/mail/u/0/?hd=princeton.edu",
    iconUrl: "https://www.google.com/s2/favicons?sz=64&domain=mail.google.com",
  },
  {
    id: "gcal",
    name: "GCal",
    url: "https://calendar.google.com/",
    iconUrl: "https://cdn.prod.website-files.com/6737568a4ac417efeb387e5a/677d0808823113a719375051_5fae8e035b232cb287900b91_Calendar_Product_Icon.svg",
  },
  {
    id: "tigerhub",
    name: "TigerHub",
    url: "https://tigerhub.princeton.edu/",
    iconUrl: "https://www.google.com/s2/favicons?sz=64&domain=tigerhub.princeton.edu",
  },
  {
    id: "tigersnatch",
    name: "TigerSnatch",
    url: "https://tigersnatch.com/",
    iconUrl: "https://www.google.com/s2/favicons?sz=64&domain=tigersnatch.com",
  },
];

const MAX_LINKS = 10;

const SIMPLE_ICON_BY_NAME: Record<string, string> = {
  canvas: "https://www.instructure.com/sites/default/files/image/2025-07/canvas_icon_color_rgb.png",
  gmail: "https://www.google.com/s2/favicons?sz=64&domain=mail.google.com",
  gcal: "https://cdn.prod.website-files.com/6737568a4ac417efeb387e5a/677d0808823113a719375051_5fae8e035b232cb287900b91_Calendar_Product_Icon.svg",
  calendar: "https://cdn.prod.website-files.com/6737568a4ac417efeb387e5a/677d0808823113a719375051_5fae8e035b232cb287900b91_Calendar_Product_Icon.svg",
  "google calendar": "https://cdn.prod.website-files.com/6737568a4ac417efeb387e5a/677d0808823113a719375051_5fae8e035b232cb287900b91_Calendar_Product_Icon.svg",
  gradescope: "https://www.google.com/s2/favicons?sz=64&domain=gradescope.com",
  youtube: "https://www.google.com/s2/favicons?sz=64&domain=youtube.com",
};

const getAvatarUrl = (name: string) =>
  `https://ui-avatars.com/api/?name=${encodeURIComponent(
    name
  )}&background=random&color=fff&size=64&rounded=true`;

const getFaviconUrl = (link: Pick<LinkItem, "name" | "url" | "iconUrl">) => {
  if (link.iconUrl) return link.iconUrl;
  const normalizedName = link.name.trim().toLowerCase();
  if (normalizedName in SIMPLE_ICON_BY_NAME) {
    return SIMPLE_ICON_BY_NAME[normalizedName];
  }
  return getAvatarUrl(link.name);
};

const sanitizeUrl = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) return "";
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed;
  }
  return `https://${trimmed}`;
};

const TigerLinks: React.FC = () => {
  const storage = useStorage();
  const [customLinks, setCustomLinks] = useState<SavedLink[]>([]);
  const [removedQuickLinks, setRemovedQuickLinks] = useState<string[]>([]);
  const [isAdding, setIsAdding] = useState(false);
  const [nameInput, setNameInput] = useState("");
  const [urlInput, setUrlInput] = useState("");
  const [error, setError] = useState("");
  const [draggingLinkId, setDraggingLinkId] = useState<string | null>(null);
  const [isOverTrash, setIsOverTrash] = useState(false);

  useEffect(() => {
    const saved = storage.getLocalStorage(StorageKeys.CUSTOM_LINKS);
    if (!saved) return;

    try {
      const parsed = JSON.parse(saved) as SavedLink[];
      if (Array.isArray(parsed)) {
        setCustomLinks(parsed.slice(0, MAX_LINKS));
      }
    } catch {
      setCustomLinks([]);
    }
  }, []);

  useEffect(() => {
    const saved = storage.getLocalStorage(StorageKeys.REMOVED_QUICK_LINKS);
    if (!saved) return;

    try {
      const parsed = JSON.parse(saved) as string[];
      if (Array.isArray(parsed)) {
        setRemovedQuickLinks(parsed.filter((id): id is string => typeof id === "string"));
      }
    } catch {
      setRemovedQuickLinks([]);
    }
  }, []);

  useEffect(() => {
    storage.setLocalStorage(StorageKeys.CUSTOM_LINKS, JSON.stringify(customLinks));
  }, [customLinks]);

  useEffect(() => {
    storage.setLocalStorage(StorageKeys.REMOVED_QUICK_LINKS, JSON.stringify(removedQuickLinks));
  }, [removedQuickLinks]);

  const links = useMemo(
    () =>
      [
        ...TIGER_LINKS.filter((link) => !removedQuickLinks.includes(link.id)),
        ...customLinks.map((link) => ({ ...link, removable: true })),
      ].slice(0, MAX_LINKS),
    [customLinks, removedQuickLinks]
  );

  const handleAddLink = (e: React.FormEvent) => {
    e.preventDefault();
    if (links.length >= MAX_LINKS) {
      setError("Maximum of 10 links reached.");
      return;
    }

    const safeName = nameInput.trim();
    const safeUrl = sanitizeUrl(urlInput);

    if (!safeName || !safeUrl) {
      setError("Name and URL are required.");
      return;
    }

    try {
      const parsedUrl = new URL(safeUrl);
      const isDuplicate = links.some(
        (link) => link.url.toLowerCase() === safeUrl.toLowerCase() || link.name.toLowerCase() === safeName.toLowerCase()
      );
      if (isDuplicate) {
        setError("That link already exists.");
        return;
      }

      const newLink: SavedLink = {
        id: `custom-${Date.now()}`,
        name: safeName,
        url: parsedUrl.toString(),
      };

      setCustomLinks((prev) => [...prev, newLink]);
      setNameInput("");
      setUrlInput("");
      setError("");
      setIsAdding(false);
    } catch {
      setError("Please enter a valid URL.");
    }
  };

  const handleRemoveLink = (id: string) => {
    setCustomLinks((prev) => prev.filter((link) => link.id !== id));
  };

  const handleTrashDrop = (id: string) => {
    if (!id) return;
    const customExists = customLinks.some((link) => link.id === id);
    if (customExists) {
      handleRemoveLink(id);
    } else {
      setRemovedQuickLinks((prev) => (prev.includes(id) ? prev : [...prev, id]));
    }
  };

  return (
    <div className={`tiger-links-wrap ${draggingLinkId ? "is-dragging" : ""}`}>
      <div className="tiger-links-header">
        <h3 className="tiger-links-title">Quick Links</h3>
        <div className="tiger-links-actions">
          <span className="tiger-links-count">{links.length}/{MAX_LINKS}</span>
          <button
            type="button"
            className={`tiger-links-add-btn ${isAdding ? "is-active" : ""}`}
            onClick={() => {
              setIsAdding((prev) => !prev);
              setError("");
            }}
          >
            {isAdding ? "Cancel" : "+ Add Link"}
          </button>
        </div>
      </div>

      <div className={`tiger-links-form-shell ${isAdding ? "is-open" : ""}`}>
        <form className="tiger-links-form" onSubmit={handleAddLink}>
          <input
            className="tiger-links-input"
            type="text"
            placeholder="Link name"
            value={nameInput}
            onChange={(e) => setNameInput(e.target.value)}
          />
          <input
            className="tiger-links-input"
            type="text"
            placeholder="example.com"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
          />
          <button className="tiger-links-save-btn" type="submit">
            Add
          </button>
        </form>
      </div>

      <div
        className={`tiger-links-error ${error ? "is-visible" : ""}`}
        role="alert"
        aria-live="polite"
      >
        {error}
      </div>

      <div className={`tiger-links-container ${draggingLinkId ? "is-dragging" : ""}`}>
        {links.map((link) => (
          <div
            key={link.id}
            className={`tiger-link-item-wrap ${draggingLinkId === link.id ? "is-dragging" : ""}`}
            draggable
            onDragStart={(e) => {
              e.dataTransfer.effectAllowed = "move";
              e.dataTransfer.setData("text/plain", link.id);
              setDraggingLinkId(link.id);
            }}
            onDragEnd={() => {
              setDraggingLinkId(null);
              setIsOverTrash(false);
            }}
          >
            <a
              href={link.url}
              className="tiger-link-item"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img
                className="tiger-link-icon"
                src={getFaviconUrl(link)}
                alt={`${link.name} icon`}
                loading="lazy"
                onError={(e) => {
                  const fallback = getAvatarUrl(link.name);
                  if (e.currentTarget.src !== fallback) {
                    e.currentTarget.src = fallback;
                  }
                }}
              />
              <div className="tiger-link-name">{link.name}</div>
            </a>
            <button
              type="button"
              className="tiger-link-remove-btn"
              onClick={() => handleTrashDrop(link.id)}
              aria-label={`Remove ${link.name}`}
              title="Remove link"
            >
              ×
            </button>
          </div>
        ))}
      </div>

      <div
        className={`tiger-links-trash-zone ${draggingLinkId ? "is-visible" : ""} ${
          isOverTrash ? "is-over" : ""
        }`}
        onDragOver={(e) => {
          if (!draggingLinkId) return;
          e.preventDefault();
          setIsOverTrash(true);
        }}
        onDragLeave={() => setIsOverTrash(false)}
        onDrop={(e) => {
          if (!draggingLinkId) return;
          e.preventDefault();
          const droppedId = e.dataTransfer.getData("text/plain") || draggingLinkId;
          handleTrashDrop(droppedId);
          setDraggingLinkId(null);
          setIsOverTrash(false);
        }}
      >
        <FiTrash2 />
      </div>
    </div>
  );
};

export default TigerLinks;
