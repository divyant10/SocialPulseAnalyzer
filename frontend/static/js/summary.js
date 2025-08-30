document.addEventListener("DOMContentLoaded", () => {
  const summaryAudio = document.getElementById("summaryAudio");
  let redirectTimeout;
  if (summaryAudio && summaryAudio.src) {
    summaryAudio.play().catch(e => console.error("Audio playback failed:", e));
  } else {
    console.warn("Summary audio element or source not found.");
  }

  
  const REDIRECT_DELAY_MS = 60 * 1000; 
  redirectTimeout = setTimeout(() => {
    if (summaryAudio) {
      summaryAudio.pause();
      summaryAudio.currentTime = 0;
    }
    window.location.href = "/dashboard";
    console.log("Redirecting to dashboard after 1 minute.");
  }, REDIRECT_DELAY_MS);

  document.addEventListener("click", () => {
    if (summaryAudio && !summaryAudio.paused) {
      summaryAudio.pause();
      summaryAudio.currentTime = 0; 
      console.log("Audio stopped by user click.");
    }
    clearTimeout(redirectTimeout);
    window.location.href = "/dashboard";
    console.log("Redirecting to dashboard immediately due to user click.");
  });
  if (summaryAudio) {
    summaryAudio.addEventListener('ended', () => {
      clearTimeout(redirectTimeout); 
      console.log("Audio finished playing. Auto-redirect timer cleared.");
    });
  }
});