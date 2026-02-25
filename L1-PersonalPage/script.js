function updateTime() {
    const now = new Date();
    
    // Time formatting
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    
    document.getElementById('clock').textContent = `${hours}:${minutes}:${seconds}`;
    
    // Date formatting
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    document.getElementById('date').textContent = now.toLocaleDateString('en-US', options);
}

// Initial call
updateTime();

// Update every second
setInterval(updateTime, 1000);

// Subtle Hover Effect
const card = document.getElementById('main-card');
document.addEventListener('mousemove', (e) => {
    if (!card) return;
    
    const { clientX, clientY } = e;
    const { innerWidth, innerHeight } = window;
    
    const xRotation = ((clientY / innerHeight) - 0.5) * 10;
    const yRotation = ((clientX / innerWidth) - 0.5) * -10;
    
    card.style.transform = `perspective(1000px) rotateX(${xRotation}deg) rotateY(${yRotation}deg) translateY(-5px)`;
});

// Reset tilt on mouse out
document.addEventListener('mouseleave', () => {
    card.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0)`;
});
