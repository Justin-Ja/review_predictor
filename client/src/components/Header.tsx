import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import InfoIcon from '@mui/icons-material/Info';
import HomeIcon from '@mui/icons-material/Home';
import NavButton from './NavButton';

function Header() {
    const navigate = useNavigate();
    const location = useLocation();
    
    const isActive = (path: string) => {
      return location.pathname === path;
    };
    
    return (
      <AppBar position="static" color="primary" sx={{ height: '60px' }}>
        <Toolbar>
          <Box>
            <NavButton 
                color="inherit"
                startIcon={<HomeIcon />}
                onClick={() => navigate('/')}
                text="Home"
                isActive={isActive('/')}
            />
            <NavButton 
                color="inherit"
                startIcon={<InfoIcon />}
                onClick={() => navigate('/info')}
                text="Info"
                isActive={isActive('/info')}
            />
          </Box>
        </Toolbar>
      </AppBar>
    );
  }
  
  export default Header;
