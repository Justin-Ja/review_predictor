import React from 'react'

interface StarCounterProps {
    selectedStars: number;
    setSelectedStars: (value: number) => void
}

const StarCounter: React.FC<StarCounterProps> = ({selectedStars, setSelectedStars}) => {
    const handleStarClick = (starCount: any) => {
        setSelectedStars(starCount);
      };

    return (
        <div>hello theressdf</div>
    )
}

export default StarCounter