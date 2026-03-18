import React from "react";
import { WidgetRow } from "./widget/WidgetRow";

type Event = {
  title: string;
  time: string;
  location: string;
  hasFood: boolean;
};

const MOCK_EVENTS: Event[] = [
  { title: "CS Department Study Break", time: "8:00 PM", location: "CS Building 104", hasFood: true },
  { title: "USG Movie Night", time: "9:30 PM", location: "Frist Film/Performance Theatre", hasFood: true },
  { title: "Guest Lecture: AI in 2026", time: "4:30 PM", location: "McCosh 50", hasFood: false },
];

const CampusEvents: React.FC = () => {
  return (
    <div className="widget-card events-widget">
      <div className="widget-header centered">
        <h3 className="bold" style={{ marginBottom: "16px" }}>
          Campus <mark>Events</mark>
        </h3>
      </div>
      <div className="widget-content">
        {MOCK_EVENTS.map((event, i) => (
          <WidgetRow key={i} props={{ index: i, data: MOCK_EVENTS }}>
            <div style={{ display: "flex", flexDirection: "column", width: "100%", gap: "4px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <h4 className="bold" style={{ margin: 0, fontSize: "15px" }}>{event.title}</h4>
                {event.hasFood && <span title="Free Food!" style={{ fontSize: "16px" }}>🍕</span>}
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px", color: "#a1a1a6" }}>
                <span>{event.time}</span>
                <span>{event.location}</span>
              </div>
            </div>
          </WidgetRow>
        ))}
      </div>
    </div>
  );
};

export default CampusEvents;
