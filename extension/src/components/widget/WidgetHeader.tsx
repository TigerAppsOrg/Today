import React from "react";

type HeaderProps = {
  title: string;
};

const WidgetHeader = (props: HeaderProps) => {
  return (
    <div className="widget-header">
      <h3 style={{ fontWeight: "600", margin: 0, paddingBottom: "16px" }}>{props.title}</h3>
    </div>
  );
};

export default WidgetHeader;
