import React from "react";

type RowData = {
  index: number;
  data: any[];
};

type RowProps = {
  children: React.ReactNode;
  props: RowData;
};

export const WidgetRow: React.FC<RowProps> = ({ children, props }) => {
  const { index, data } = props;

  return (
    <div
      className={index >= data.length - 1 ? "divider no-divider" : "divider"}
      key={index}
    >
      <div className="row-content">{children}</div>
    </div>
  );
};
