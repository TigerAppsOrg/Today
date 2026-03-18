import React from "react";
import { WidgetRow } from "./widget/WidgetRow";

type AppItem = {
  name: string;
  description: string;
  url: string;
};

const TIGER_APPS: AppItem[] = [
  { name: "TigerSnatch", description: "Get notified when a course opens up", url: "https://tigersnatch.com/" },
  { name: "TigerPath", description: "Four-year course planner", url: "https://www.tigerpath.io/" },
  { name: "Princeton Courses", description: "Course reviews and ratings", url: "https://www.princetoncourses.com/" },
  { name: "ReCal", description: "Course scheduling and calendar", url: "https://recal.app/" },
  { name: "TigerDraw", description: "Room draw guide and floor plans", url: "https://tigerdraw.com/" },
];

const TigerAppsWidget: React.FC = () => {
  return (
    <div className="widget-card tiger-apps-widget">
      <div className="widget-header centered">
        <h3 className="bold" style={{ marginBottom: "16px" }}>
          Tiger<mark>Apps</mark>
        </h3>
      </div>
      <div className="widget-content">
        {TIGER_APPS.map((app, i) => (
          <WidgetRow key={i} props={{ index: i, data: TIGER_APPS }}>
            <a href={app.url} target="_blank" rel="noopener noreferrer" style={{ textDecoration: "none", color: "inherit", display: "flex", flexDirection: "column", width: "100%", gap: "4px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <h4 className="bold" style={{ margin: 0, fontSize: "16px" }}>{app.name}</h4>
              </div>
              <div style={{ fontSize: "14px", color: "#a1a1a6" }}>
                {app.description}
              </div>
            </a>
          </WidgetRow>
        ))}
      </div>
    </div>
  );
};

export default TigerAppsWidget;
