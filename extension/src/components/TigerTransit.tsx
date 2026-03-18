import React, { useState } from "react";
import { WidgetRow } from "./widget/WidgetRow";

type TransitStop = {
  name: string;
  routes: { route: string; time: string; color: string }[];
};

const MOCK_DATA: TransitStop[] = [
  {
    name: "Frist Campus Center",
    routes: [
      { route: "Route 1", time: "2 min", color: "#E77500" },
      { route: "Route 2", time: "5 min", color: "#005596" },
      { route: "Route 4", time: "12 min", color: "#4A90E2" },
    ],
  },
];

const TigerTransit: React.FC = () => {
  const [stop] = useState<TransitStop>(MOCK_DATA[0]);

  return (
    <div className="widget-card transit-widget">
      <div className="widget-header centered">
        <h3 className="bold" style={{ marginBottom: "12px" }}>
          Tiger<mark>Transit</mark>
        </h3>
        <div style={{ color: "#a1a1a6", fontSize: "14px", marginBottom: "16px" }}>
          {stop.name}
        </div>
      </div>
      <div className="widget-content">
        {stop.routes.map((route, i) => (
          <WidgetRow key={i} props={{ index: i, data: stop.routes }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", width: "100%" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <div style={{ width: "12px", height: "12px", borderRadius: "50%", backgroundColor: route.color }}></div>
                <h4 className="bold" style={{ margin: 0, fontSize: "16px" }}>{route.route}</h4>
              </div>
              <div style={{ fontWeight: "600", color: "#f5f5f7" }}>{route.time}</div>
            </div>
          </WidgetRow>
        ))}
      </div>
    </div>
  );
};

export default TigerTransit;
