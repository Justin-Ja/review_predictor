import React from 'react';

interface TitleProps {
  text: string;
  color?: "black" | "blue";
}

function Title({ text, color = "black"}: TitleProps) {
  const colorClasses = {
    black: "text-black",
    blue: "text-blue-600",
  };
  
  return (
    <div className="text-center">
      <h1 className={`text-4xl font-bold ${colorClasses[color]}`}>
        {text}
      </h1>
    </div>
  );
}

export default Title;
