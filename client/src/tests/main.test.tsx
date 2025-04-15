import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Main from '../pages/main';
import '@testing-library/jest-dom';
import App from '../App';
import { MemoryRouter } from 'react-router-dom';

beforeEach(() => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        json: () =>
          Promise.resolve({
            text: 'This is a sample review.',
            score: 4,
            pred_score: 3.5,
          }),
      })
    ) as jest.Mock;
  });

  const renderWithRouter = () => {
    return render(
      <MemoryRouter>
        <Main />
      </MemoryRouter>
    );
  };
  
  
  describe('Main Page', () => {
    test('renders title and header', async () => {
      renderWithRouter();
      expect(await screen.findByText('Review Guessing Game')).toBeInTheDocument();
    });

    test('loads and displays review text', async () => {
      renderWithRouter();
      const reviewText = await screen.findByText(/This is a sample review/i);
      expect(reviewText).toBeInTheDocument();
    });
    
    test('clicks the 4-star and updates selection', async () => {
      renderWithRouter();
    
      const star4 = await screen.findByTestId('star-4');
      fireEvent.click(star4);
    
      const submitButton = await screen.findByTestId('game-button');
      expect(submitButton).toBeEnabled();
    });
    
    test('disables submit button when no star selected', async () => {
      renderWithRouter();
      const submitButton = await screen.findByTestId("game-button");
      expect(submitButton).toBeDisabled();
    });
    
    
    test('shows results after submitting a guess', async () => {
      renderWithRouter();
      const star4 = await screen.findByTestId('star-4');
      fireEvent.click(star4);
  
      const submitButton = await screen.findByTestId('game-button');
      fireEvent.click(submitButton);
  
      await waitFor(() => {
        expect(screen.getByText(/The model predicted a/i)).toBeInTheDocument();
        expect(screen.getByText(/The actual review was given/i)).toBeInTheDocument();
      });
    });
    
    test('shows Next button after guess', async () => {
      renderWithRouter();
      const star4 = await screen.findByTestId('star-4');
      fireEvent.click(star4);

      const submitButton = await screen.findByTestId('game-button');
      fireEvent.click(submitButton);

      await screen.findByText(/Start a new Round/i);
    });
  });  