import React from "react";

type Props = {
  title: string;
};

const ComingSoonWidget: React.FC<Props> = ({ title }) => {
  return (
    <div className="widget-card coming-soon-widget" style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", minHeight: "200px" }}>
      <div className="widget-header centered">
        <h3 className="bold" style={{ marginBottom: "16px", color: "#a1a1a6" }}>
          {title}
        </h3>
      </div>
      <div style={{ fontSize: "24px", fontWeight: "bold", color: "#E77500", opacity: 0.8 }}>
        COMING SOON
      </div>
    </div>
  );
};

export default ComingSoonWidget;
