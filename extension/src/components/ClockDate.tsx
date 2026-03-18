import React, { useEffect, useState } from "react";
import { useTime } from "../context/TimeContext";
import "./ClockDate.css";

const ClockDate: React.FC = () => {
  const time = useTime();
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const timeString = currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <div className="clock-date-container">
      <div className="clock-time">{timeString}</div>
      <div className="clock-date">{time.dateString}</div>
    </div>
  );
};

export default ClockDate;
