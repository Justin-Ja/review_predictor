import React from 'react'
import { Rating } from '@mui/material';
import StarIcon from '@mui/icons-material/Star';
import StarBorderIcon from '@mui/icons-material/StarBorder';

interface StarCounterProps {
    selectedStars: number;
    setSelectedStars: (value: number) => void
    hasUserGuessed: boolean
}

const StarCounter: React.FC<StarCounterProps> = ({selectedStars, setSelectedStars, hasUserGuessed}) => {
  const handleChange = (event: React.SyntheticEvent, newValue: number | null) => {
    if (newValue !== null) {
      setSelectedStars(newValue);
    }
  };

  //Adds a testid to each star
  const CustomIconContainer = ({ value, ...other }: { value: number }) => (
    <span data-testid={`star-${value}`} {...other} />
  );

  return (
    <Rating
      data-testid="star-rating"
      name="star-rating"
      value={selectedStars}
      onChange={handleChange}
      precision={1}
      size="large"
      readOnly={hasUserGuessed}
      icon={<StarIcon fontSize="inherit" sx={{ color: '#FFD700' }} />}
      emptyIcon={<StarBorderIcon fontSize="inherit" />}
      IconContainerComponent={CustomIconContainer}
      sx={{
        '& .MuiRating-iconFilled': {
          color: '#FFD700',
        },
        '& .MuiRating-iconHover': {
          color: '#FFCC00',
        },
        '& .MuiRating-iconEmpty': {
          color: '#BDBDBD',
        }
      }}
    />
  );
};


export default StarCounter