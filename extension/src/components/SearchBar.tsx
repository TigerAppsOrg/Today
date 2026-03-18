import React, { useEffect, useState } from "react";
import { FiSearch } from "react-icons/fi";
import "./SearchBar.css";
import { StorageKeys, useStorage } from "../context/StorageContext";

const MAX_HISTORY_ITEMS = 8;

const SearchBar: React.FC = () => {
  const storage = useStorage();
  const [query, setQuery] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);

  useEffect(() => {
    const savedHistory = storage.getLocalStorage(StorageKeys.SEARCH_HISTORY);
    if (!savedHistory) return;

    try {
      const parsed = JSON.parse(savedHistory);
      if (Array.isArray(parsed)) {
        setSearchHistory(parsed.filter((item) => typeof item === "string").slice(0, MAX_HISTORY_ITEMS));
      }
    } catch {
      setSearchHistory([]);
    }
  }, [storage]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    const updatedHistory = [
      trimmedQuery,
      ...searchHistory.filter((item) => item.toLowerCase() !== trimmedQuery.toLowerCase()),
    ].slice(0, MAX_HISTORY_ITEMS);
    setSearchHistory(updatedHistory);
    storage.setLocalStorage(StorageKeys.SEARCH_HISTORY, JSON.stringify(updatedHistory));

    const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(trimmedQuery)}`;
    window.location.href = searchUrl;
  };

  return (
    <div className={`search-container ${isFocused ? "focused" : ""}`}>
      <form className="search-form" onSubmit={handleSearch}>
        <FiSearch className="search-icon" />
        <input
          type="text"
          className="search-input"
          placeholder="Search the web..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          autoFocus
        />
      </form>
    </div>
  );
};

export default SearchBar;
