import { useState, useEffect, useRef } from "react";
import Dropdown from "react-bootstrap/Dropdown";
import React from "react";
import { StorageKeys, useStorage } from "../context/StorageContext";
import { useTime, Hours, Days } from "../context/TimeContext";
import { useData } from "../context/DataContext";
import { WidgetRow } from "./widget/WidgetRow";
import { EventTypes, useMixpanel } from "../context/MixpanelContext";

type MealSession = "Breakfast" | "Lunch" | "Dinner";

type MealItem = {
  cat: string;
  items: string;
};

type DiningHall = {
  key: string;
  label: string;
};

const DINING_HALLS: DiningHall[] = [
  { key: "Yeh/NCW", label: "Yeh/NCW" },
  { key: "Forbes", label: "Forbes" },
  { key: "Roma", label: "Roma" },
  { key: "Whitman", label: "Whitman" },
  { key: "Center for Jewish Life", label: "CJL" },
];
const DEFAULT_DHALL = DINING_HALLS[0].key;

function DHallTable() {
  const DROPDOWN_CLOSE_MS = 180;
  const storage = useStorage();
  const time = useTime();
  const data = useData();
  const mixpanel = useMixpanel();
  const closeTimerRef = useRef<number | null>(null);
  const validResults = DINING_HALLS.map((diningHall) => diningHall.key);

  const [college, setCollege] = useState(
    storage.getLocalStorageDefault(
      StorageKeys.DHALL,
      DEFAULT_DHALL,
      validResults
    )
  );
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isDropdownClosing, setIsDropdownClosing] = useState(false);

  useEffect(() => {
    storage.setLocalStorage(StorageKeys.DHALL, college);
  }, [college]);

  useEffect(() => {
    return () => {
      if (closeTimerRef.current) {
        window.clearTimeout(closeTimerRef.current);
      }
    };
  }, []);

  const currentDay = time.dayPrinceton;
  const currentHour = time.currentHourPrinceton;

  let meal: MealSession = "Breakfast";
  if ((currentDay === Days.Saturday || currentDay === Days.Sunday) && currentHour < Hours._10AM) {
    meal = "Lunch";
  } else if (Hours._10AM <= currentHour && currentHour < Hours._2PM) {
    meal = "Lunch";
  } else if (Hours._2PM <= currentHour && currentHour < Hours._12AM) {
    meal = "Dinner";
  }

  const priority = [
    "Entree",
    "Main Entree",
    "Early Entree",
    "Vegetarian & Vegan Entree",
    "Vegan/Vegetarian",
    "Pasta",
    "Specialty Bars",
    "Breakfast Bars",
    "On the Side",
    "Sides",
    "Grill",
    "Action Station",
    "Pasta Station",
    "Desserts",
    "Breakfast Cereal",
    "Composed Salads",
    "Soup of the Day",
    "Soups",
    "Salads",
  ];

  const dhallData = (data?.dhall?.[college]?.[meal] as Record<string, string[]> | null) || null;
  const selectedHallLabel =
    DINING_HALLS.find((diningHall) => diningHall.key === college)?.label ??
    DEFAULT_DHALL;
  const orderedData: MealItem[] = [];

  if (dhallData) {
    const priorityCategories = priority.filter((category) => category in dhallData);
    const remainingCategories = Object.keys(dhallData).filter(
      (category) => !priority.includes(category)
    );
    const orderedCategories = [...priorityCategories, ...remainingCategories];

    orderedCategories.forEach((category) => {
      orderedData.push({
        cat: category,
        items: dhallData[category].join(", "),
      });
    });
  }

  const rows = orderedData.map((item, i) => {
    return (
      <WidgetRow key={i} props={{ index: i, data: orderedData }}>
        <h4 className="bold">{item.cat}</h4> {item.items}
      </WidgetRow>
    );
  });

  if (rows.length === 0) {
    rows.push(
      <WidgetRow key={0} props={{ index: 0, data: [] }}>
        <h4 style={{ textAlign: "center", marginTop: 36 }}>
          Dining Hall Closed
        </h4>
      </WidgetRow>
    );
  }

  return (
    <div className="widget-card dining-hall">
      <div className="widget-header dhall-header">
        <div className="dhall-title-row">
          <h3 className="bold dhall-title">
            What's for <mark>{meal}</mark> at
          </h3>
          <Dropdown
            className={`dhall-dropdown ${isDropdownClosing ? "is-closing" : ""}`}
            show={isDropdownOpen}
            onToggle={(nextShow) => {
              if (closeTimerRef.current) {
                window.clearTimeout(closeTimerRef.current);
                closeTimerRef.current = null;
              }

              if (nextShow) {
                setIsDropdownClosing(false);
                setIsDropdownOpen(true);
                return;
              }

              setIsDropdownClosing(true);
              closeTimerRef.current = window.setTimeout(() => {
                setIsDropdownOpen(false);
                setIsDropdownClosing(false);
                closeTimerRef.current = null;
              }, DROPDOWN_CLOSE_MS);
            }}
            onSelect={(e) => {
              const dhall = e || DEFAULT_DHALL;
              setCollege(dhall);
              mixpanel.trackEvent(EventTypes.DHALL_CHANGE, dhall);
            }}
          >
            <Dropdown.Toggle className="dhall-dropdown-toggle">
              {selectedHallLabel}
            </Dropdown.Toggle>
            <Dropdown.Menu
              className={`dhall-dropdown-menu ${isDropdownClosing ? "is-closing" : ""}`}
            >
              {DINING_HALLS.map((diningHall) => (
                <Dropdown.Item
                  key={diningHall.key}
                  eventKey={diningHall.key}
                  className={diningHall.key === college ? "is-active" : ""}
                >
                  {diningHall.label}
                </Dropdown.Item>
              ))}
            </Dropdown.Menu>
          </Dropdown>
        </div>
      </div>
      <div className="widget-content">
        {rows}
      </div>
    </div>
  );
}

export default DHallTable;
