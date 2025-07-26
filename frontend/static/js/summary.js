document.addEventListener("DOMContentLoaded", () => {
  const summaryAudio = document.getElementById("summaryAudio");
  let redirectTimeout;

  // --- 1. Play audio automatically (handled by summary.html's <audio autoplay> or .play() on load) ---
  // If autoplay isn't reliable, you can ensure it starts here:
  if (summaryAudio && summaryAudio.src) {
    summaryAudio.play().catch(e => console.error("Audio playback failed:", e));
  } else {
    console.warn("Summary audio element or source not found.");
  }

  // --- 2. Stop audio and redirect after 1 minute ---
  const REDIRECT_DELAY_MS = 60 * 1000; // 1 minute (60 seconds)
  redirectTimeout = setTimeout(() => {
    // Ensure audio is stopped before redirecting
    if (summaryAudio) {
      summaryAudio.pause();
      summaryAudio.currentTime = 0;
    }
    window.location.href = "/dashboard";
    console.log("Redirecting to dashboard after 1 minute.");
  }, REDIRECT_DELAY_MS);

  // --- 3. Stop audio and redirect immediately if user clicks anywhere ---
  document.addEventListener("click", () => {
    // Stop the audio if it's playing
    if (summaryAudio && !summaryAudio.paused) {
      summaryAudio.pause();
      summaryAudio.currentTime = 0; // Rewind to start
      console.log("Audio stopped by user click.");
    }
    // Clear the auto-redirect timer as user has interacted and is being redirected
    clearTimeout(redirectTimeout);
    // Redirect to dashboard immediately on click
    window.location.href = "/dashboard";
    console.log("Redirecting to dashboard immediately due to user click.");
  });

  // Optional: You might also want to clear the timeout if the audio finishes naturally
  if (summaryAudio) {
    summaryAudio.addEventListener('ended', () => {
      clearTimeout(redirectTimeout); // Clear the auto-redirect timer
      console.log("Audio finished playing. Auto-redirect timer cleared.");
      // If you want it to redirect immediately *after* audio finishes,
      // and not wait for the 1-min timer, you could add:
      // window.location.href = "/dashboard";
      // But the requirement was 1 min redirect *regardless* of audio status.
    });
  }
});