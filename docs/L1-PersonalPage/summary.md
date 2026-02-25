# Project Summary: L1-PersonalPage

## Objective
To create a high-end, responsive personal landing page that displays the user's name and current time using modern web technologies.

## Technical Implementation
- **Layout**: Centered flexbox container with a responsive glass-style card.
- **Styling**: Used CSS variables for theme management, custom Google Fonts (Outfit), and keyframe animations for background blobs.
- **Interactivity**: 
    - Implemented a `setInterval` function in JS to update the time every second.
    - Added an event listener for `mousemove` to calculate and apply CSS `transform: rotateX/rotateY` for the 3D tilt effect.

## Challenges & Solutions
- **Consistency**: Ensured the clock updates smoothly without layout shifts by using monospaced numeric fonts.
- **Aesthetics**: Achieved the glass effect using `backdrop-filter: blur()` and subtle borders to maintain high visual quality.

## Outcome
The project successfully meets all requirements, providing a functional, premium-looking single-page application that serves as a professional introduction for Sam WEI.
