import React from 'react'

interface StarCounterProps {
    selectedStars: number;
    setSelectedStars: (value: number) => void
}

const StarCounter: React.FC<StarCounterProps> = ({selectedStars, setSelectedStars}) => {
    const handleStarClick = (starCount: any) => {
        setSelectedStars(starCount);
      };

      //TODO: Make a better star. or import UI library (no half stars bc all labels are ints. I guess half could be allowed but it be impossible to guess 1.5 and the actual score to be 1.5)
    return (
        <>
        {[1, 2, 3, 4, 5].map((star) => (
          <span 
            key={star} 
            onClick={() => handleStarClick(star)} 
            style={{ cursor: 'pointer', color: star <= selectedStars ? 'gold' : 'gray' }}
          >
            ★
          </span>
        ))}
      </>
    )
}

export default StarCounter