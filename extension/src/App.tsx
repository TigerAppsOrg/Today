import DHallTable from "./components/DiningHalls";
import React, { useEffect, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import CampusEvents from "./components/CampusEvents";
import { EventTypes, useMixpanel } from "./context/MixpanelContext";
import { StorageKeys, useStorage } from "./context/StorageContext";
import "react-sliding-pane/dist/react-sliding-pane.css";
import "./App.css";
import WeatherTable from "./components/Weather";
import SearchBar from "./components/SearchBar";
import TigerLinks from "./components/TigerLinks";
import TigerTransit from "./components/TigerTransit";
import ClockDate from "./components/ClockDate";
import Chat from "./components/Chat";
import TigerAppsWidget from "./components/TigerAppsWidget";
import ComingSoonWidget from "./components/ComingSoonWidget";
import SettingsModal from "./components/SettingsModal";
import { FiMove, FiSettings } from "react-icons/fi";
import TodayNewsWidget from "./components/TodayNewsWidget";

const AVAILABLE_WIDGETS = [
  { id: "dhall", name: "Dinner" },
  { id: "weather", name: "Weather" },
  { id: "today", name: "Today News" },
];

const DEFAULT_WIDGETS = ["dhall", "weather", "today"];
const ALLOWED_WIDGET_IDS = new Set(AVAILABLE_WIDGETS.map((widget) => widget.id));
const ALL_WIDGET_COLUMNS = ["left", "middle", "right"] as const;
const MIN_WIDGET_COLUMNS = 2;
const MAX_WIDGET_COLUMNS = 3;

type WidgetColumn = (typeof ALL_WIDGET_COLUMNS)[number];
type WidgetLayout = Record<WidgetColumn, string[]>;

const getActiveColumns = (columnCount: number): WidgetColumn[] => {
  const safeCount = Math.max(MIN_WIDGET_COLUMNS, Math.min(MAX_WIDGET_COLUMNS, columnCount));
  return safeCount === 2 ? ["left", "right"] : ["left", "middle", "right"];
};

const createEmptyLayout = (): WidgetLayout => ({
  left: [],
  middle: [],
  right: [],
});

const buildDefaultLayout = (widgets: string[], columnCount: number): WidgetLayout => {
  const layout = createEmptyLayout();
  const activeColumns = getActiveColumns(columnCount);
  widgets.forEach((widgetId, index) => {
    const targetColumn = activeColumns[index % activeColumns.length];
    layout[targetColumn].push(widgetId);
  });
  return layout;
};

const normalizeWidgetLayout = (
  layout: Partial<WidgetLayout> | null,
  activeWidgets: string[],
  columnCount: number
): WidgetLayout => {
  const activeColumns = getActiveColumns(columnCount);
  const activeSet = new Set(activeWidgets);
  const seen = new Set<string>();

  const normalized = createEmptyLayout();

  // Keep explicit user ordering for currently active columns.
  activeColumns.forEach((column) => {
    normalized[column] = (layout?.[column] || []).filter((id) => {
      if (!activeSet.has(id) || seen.has(id)) return false;
      seen.add(id);
      return true;
    });
  });

  // Re-home widgets from inactive columns + any new/missing widgets.
  const extras = ALL_WIDGET_COLUMNS.filter((column) => !activeColumns.includes(column))
    .flatMap((column) => layout?.[column] || [])
    .filter((id) => activeSet.has(id) && !seen.has(id));

  activeWidgets.forEach((id) => {
    if (!seen.has(id)) {
      extras.push(id);
    }
  });

  extras.forEach((id) => {
    if (seen.has(id)) return;
    const targetColumn = activeColumns.reduce((shortest, candidate) =>
      normalized[candidate].length < normalized[shortest].length ? candidate : shortest
    );
    normalized[targetColumn].push(id);
    seen.add(id);
  });

  return normalized;
};

function App() {
  const storage = useStorage();
  const mixpanel = useMixpanel();

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isArrangeMode, setIsArrangeMode] = useState(false);
  const [draggingWidgetId, setDraggingWidgetId] = useState<string | null>(null);
  const [dropTargetKey, setDropTargetKey] = useState<string | null>(null);
  const [snappedWidgetId, setSnappedWidgetId] = useState<string | null>(null);
  const [columnCount, setColumnCount] = useState<2 | 3>(() => {
    const saved = storage.getLocalStorage(StorageKeys.WIDGET_COLUMN_COUNT);
    return saved === "2" ? 2 : 3;
  });
  
  const [activeWidgets, setActiveWidgets] = useState<string[]>(() => {
    const saved = storage.getLocalStorage(StorageKeys.ACTIVE_WIDGETS);
    if (!saved) return DEFAULT_WIDGETS;

    try {
      const parsed = JSON.parse(saved);
      if (!Array.isArray(parsed)) return DEFAULT_WIDGETS;
      const filtered = parsed.filter((id): id is string => typeof id === "string" && ALLOWED_WIDGET_IDS.has(id));
      return filtered;
    } catch {
      return DEFAULT_WIDGETS;
    }
  });
  const [widgetLayout, setWidgetLayout] = useState<WidgetLayout>(() => {
    const saved = storage.getLocalStorage(StorageKeys.WIDGET_LAYOUT);
    if (!saved) return buildDefaultLayout(activeWidgets, columnCount);

    try {
      const parsed = JSON.parse(saved) as Partial<WidgetLayout>;
      return normalizeWidgetLayout(parsed, activeWidgets, columnCount);
    } catch {
      return buildDefaultLayout(activeWidgets, columnCount);
    }
  });

  useEffect(() => {
    const customBackground = storage.getLocalStorage(StorageKeys.CUSTOM_BG);
    if (customBackground) {
      document.body.style.backgroundImage = `url('${customBackground}')`;
      document.body.style.backgroundSize = "cover";
      document.body.style.backgroundPosition = "center center";
      document.body.style.backgroundAttachment = "fixed";
      return;
    }
    document.body.style.backgroundImage = "";
    document.body.style.backgroundSize = "";
    document.body.style.backgroundPosition = "";
    document.body.style.backgroundAttachment = "";
  }, []);

  useEffect(() => {
    const state = storage.getLocalStorageObject();
    mixpanel.trackEvent(EventTypes.PAGE_LOAD, state);
  }, []);

  useEffect(() => {
    storage.setLocalStorage(StorageKeys.ACTIVE_WIDGETS, JSON.stringify(activeWidgets));
  }, [activeWidgets]);

  useEffect(() => {
    setWidgetLayout((prev) => normalizeWidgetLayout(prev, activeWidgets, columnCount));
  }, [activeWidgets, columnCount]);

  useEffect(() => {
    storage.setLocalStorage(StorageKeys.WIDGET_LAYOUT, JSON.stringify(widgetLayout));
  }, [widgetLayout]);

  useEffect(() => {
    storage.setLocalStorage(StorageKeys.WIDGET_COLUMN_COUNT, String(columnCount));
  }, [columnCount]);

  const toggleWidget = (id: string) => {
    if (!ALLOWED_WIDGET_IDS.has(id)) return;
    setActiveWidgets(prev => 
      prev.includes(id) ? prev.filter(w => w !== id) : [...prev, id]
    );
  };

  const renderWidget = (id: string) => {
    switch(id) {
      case "dhall": return <DHallTable />;
      case "weather": return <WeatherTable />;
      case "today": return <TodayNewsWidget />;
      case "transit": return <TigerTransit />;
      case "events": return <CampusEvents />;
      case "chat": return <div className="widget-card chat-widget" style={{ padding: 0, overflow: "hidden" }}><Chat /></div>;
      case "tigerapps": return <TigerAppsWidget />;
      case "laundry": return <ComingSoonWidget title="Laundry Status" />;
      case "printer": return <ComingSoonWidget title="Printer Status" />;
      default: return null;
    }
  };

  const moveWidget = (fromId: string, toColumn: WidgetColumn, toIndex: number) => {
    if (!fromId) return;

    setWidgetLayout((prev) => {
      const next = normalizeWidgetLayout(prev, activeWidgets, columnCount);
      let fromColumn: WidgetColumn | null = null;
      let fromIndex = -1;

      for (const column of ALL_WIDGET_COLUMNS) {
        const index = next[column].indexOf(fromId);
        if (index === -1 || fromColumn) continue;
        fromColumn = column;
        fromIndex = index;
      }

      if (!fromColumn) return next;

      next[fromColumn] = next[fromColumn].filter((id) => id !== fromId);

      let insertionIndex = Math.max(0, Math.min(toIndex, next[toColumn].length));
      if (fromColumn === toColumn && fromIndex < toIndex) {
        insertionIndex = Math.max(0, insertionIndex - 1);
      }

      next[toColumn].splice(insertionIndex, 0, fromId);
      return next;
    });
    setSnappedWidgetId(fromId);
    window.setTimeout(() => setSnappedWidgetId(null), 380);
  };

  const feedbackMailto =
    "mailto:aj5828@princeton.edu?subject=TodayCustom%20Feedback%20%28Free%20Coffee%29&body=Hey%20Khan%2C%0A%0AHere%20is%20my%20feedback%20for%20TodayCustom%3A%0A-%20%0A-%20%0A%0AThanks!";
  const activeColumns = getActiveColumns(columnCount);

  return (
    <div className="App">
      <button
        className="settings-button-top"
        onClick={() => setIsSettingsOpen(true)}
        aria-label="Open customization settings"
        title="Customize dashboard"
      >
        <FiSettings className="settings-button-icon" />
        <span>Customize</span>
      </button>

      <div className="main-content">
        <ClockDate />
        <SearchBar />
        <TigerLinks />
        
        {activeWidgets.length > 0 && (
          <div className="widget-arrange-toolbar">
            <button
              className={`arrange-button ${isArrangeMode ? "is-active" : ""}`}
              onClick={() => {
                setIsArrangeMode((prev) => !prev);
                setDraggingWidgetId(null);
                setDropTargetKey(null);
              }}
              aria-label="Rearrange widgets"
              title="Rearrange widgets"
            >
              <FiMove className="arrange-button-icon" />
              <span>{isArrangeMode ? "Done Arranging" : "Arrange Widgets"}</span>
            </button>
          </div>
        )}

        <div
          className={`command-center-widgets columns-${columnCount} ${isArrangeMode ? "is-arranging" : ""}`}
        >
          {activeColumns.map((column) => (
            <div
              key={column}
              className={[
                "widget-column",
                isArrangeMode ? "is-arranging" : "",
                dropTargetKey === `${column}:end` ? "is-drop-target" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              onDragOver={(e) => {
                if (!isArrangeMode) return;
                e.preventDefault();
                if (dropTargetKey !== `${column}:end`) {
                  setDropTargetKey(`${column}:end`);
                }
              }}
              onDrop={(e) => {
                if (!isArrangeMode) return;
                e.preventDefault();
                const draggedId = e.dataTransfer.getData("text/plain");
                moveWidget(draggedId, column, widgetLayout[column].length);
                setDraggingWidgetId(null);
                setDropTargetKey(null);
              }}
            >
              {widgetLayout[column].map((widgetId) => (
                <div
                  key={widgetId}
                  className={[
                    "widget-slot",
                    draggingWidgetId === widgetId ? "is-dragging" : "",
                    dropTargetKey === `${column}:${widgetId}` ? "is-drop-target" : "",
                    snappedWidgetId === widgetId ? "is-snapped" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  draggable={isArrangeMode}
                  onDragStart={(e) => {
                    if (!isArrangeMode) return;
                    e.dataTransfer.effectAllowed = "move";
                    e.dataTransfer.setData("text/plain", widgetId);
                    setDraggingWidgetId(widgetId);
                  }}
                  onDragOver={(e) => {
                    if (!isArrangeMode) return;
                    e.preventDefault();
                    e.stopPropagation();
                    if (dropTargetKey !== `${column}:${widgetId}`) {
                      setDropTargetKey(`${column}:${widgetId}`);
                    }
                  }}
                  onDrop={(e) => {
                    if (!isArrangeMode) return;
                    e.preventDefault();
                    e.stopPropagation();
                    const draggedId = e.dataTransfer.getData("text/plain");
                    const targetIndex = widgetLayout[column].indexOf(widgetId);
                    const slotRect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
                    const isDropAfter = e.clientY >= slotRect.top + slotRect.height / 2;
                    moveWidget(draggedId, column, targetIndex + (isDropAfter ? 1 : 0));
                    setDraggingWidgetId(null);
                    setDropTargetKey(null);
                  }}
                  onDragEnd={() => {
                    setDraggingWidgetId(null);
                    setDropTargetKey(null);
                  }}
                >
                  {isArrangeMode && (
                    <div className="widget-slot-badge">
                      <FiMove />
                      <span>Drag to move</span>
                    </div>
                  )}
                  {renderWidget(widgetId)}
                </div>
              ))}
              {isArrangeMode && widgetLayout[column].length === 0 && (
                <div className="widget-column-empty">Drop widgets here</div>
              )}
            </div>
          ))}
        </div>
      </div>

      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)} 
        activeWidgets={activeWidgets}
        toggleWidget={toggleWidget}
        availableWidgets={AVAILABLE_WIDGETS}
        columnCount={columnCount}
        onColumnCountChange={setColumnCount}
      />

      <a
        className="settings-button-top feedback-button-floating"
        href={feedbackMailto}
        aria-label="Contribute feedback by email for free coffee"
      >
        Contribute Feedback (Free Coffee on Me)
      </a>
    </div>
  );
}

export default App;
