import React from 'react';
import { Button, SxProps } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

interface GameButtonProps {
  onClick: () => void;
  isSubmit: boolean; // true for submit, false for new round
  disabled?: boolean;
}

const GameButton: React.FC<GameButtonProps> = ({ onClick, isSubmit, disabled = false }) => {
    
  const buttonStyles: SxProps = {
    minWidth: '180px',
    fontWeight: 'bold',
    textTransform: 'none',
    borderRadius: '8px',
    paddingTop: 1,
    boxShadow: 3,
    transition: 'all 0.2s ease-in-out',
    '&:hover': {
      transform: 'translateY(-2px)',
      boxShadow: 4,
    },
    bgcolor: isSubmit ? 'primary.main' : 'success.main',
  };

  return (
    <Button
      data-testid="game-button"
      variant="contained"
      color={isSubmit ? "primary" : "success"}
      onClick={onClick}
      disabled={disabled}
      startIcon={isSubmit ? <SendIcon /> : <PlayArrowIcon />}
      sx={buttonStyles}
    >
      {isSubmit ? 'Submit your guess' : 'Start a new Round'}
    </Button>
  );
};

export default GameButton;
