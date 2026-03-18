import React from "react";
import { useData } from "../context/DataContext";
import { WidgetRow } from "./widget/WidgetRow";
import { EventTypes, useMixpanel } from "../context/MixpanelContext";

type Article = {
  title: string;
  link: string;
};

const TodayNewsWidget: React.FC = () => {
  const data = useData();
  const mixpanel = useMixpanel();
  const articles: Article[] = data?.prince?.articles || [];

  const rows = articles.map((article, index) => (
    <WidgetRow key={article.link || index} props={{ index, data: articles }}>
      <a
        href={article.link}
        className="today-news-link"
        onClick={() => mixpanel.trackEvent(EventTypes.NEWS_CLICK, article)}
      >
        {article.title}
      </a>
    </WidgetRow>
  ));

  if (rows.length === 0) {
    rows.push(
      <WidgetRow key="today-news-empty" props={{ index: 0, data: [] }}>
        <div className="today-news-empty">No stories available right now.</div>
      </WidgetRow>
    );
  }

  return (
    <div className="widget-card today-news-widget">
      <div className="today-news-source" aria-label="News source">
        From The Daily Princetonian
      </div>
      <div className="widget-content">{rows}</div>
    </div>
  );
};

export default TodayNewsWidget;
