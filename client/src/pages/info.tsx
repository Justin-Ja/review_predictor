import React from 'react';
import Header from '../components/Header';
import Title from '../components/Title';
import '../styles/Info.css';

function Info(): React.ReactElement {
  return (
    <div className="App">
      <Header />
      <Title text="How to Play" />

      <div className="info-container">

        <section className="info-section">
          <h2>What is this?</h2>
          <p>
            This is a <strong>machine learning web application</strong> where you compete against a
            trained AI model. Each round you read a real review and predict the star rating it was
            given, using only the text itself. The AI has learned from thousands of reviews and picks up on subtle linguistic patterns.
            Can your human intuition beat it?
          </p>
        </section>

        <section className="info-section">
          <h2>How it Works</h2>
          <p>
            You'll be shown a review with its star rating hidden. Both you and the AI submit a
            predicted score, then the actual rating is revealed and points are awarded based on
            accuracy. Miss by more than one star and neither of you scores. The closer the guess, the more
            points you earn.
          </p>
        </section>

        <section className="info-section">
          <h2>The Playing Field</h2>
          <div className="edge-box">
            <p>
              ⚠️ <strong>Fair warning:</strong> the AI has a slight edge. To score 2 points you need
              an exact star match, but the AI only needs to land within <strong>0.34 stars</strong> of
              the true score — because it outputs a continuous number rather than a whole star.
            </p>
            <p>
              That said, you have the advantage of being a human. Unless you're not. Then skill issue.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}

export default Info;