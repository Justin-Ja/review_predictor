import React from 'react';
import { Button } from '@mui/material';

interface NavButtonProps {
    color: "inherit" | "primary" | "success" | "error" | "info" | "warning";
    startIcon: React.ReactNode;
    onClick: () => void;
    text: string;
    isActive: boolean;
}

function NavButton({ color, startIcon, onClick, text, isActive }: NavButtonProps) {
  
return (
    <Button
      color={color}
      startIcon={startIcon}
      onClick={onClick}
      sx={{
        fontWeight: isActive ? 'bold' : 'normal',
        borderBottom: isActive ? '2px solid white' : 'none',
        marginRight: 2,
        marginBottom: 0.5,
      }}
    >
      {text}
    </Button>
  );
}

export default NavButton;