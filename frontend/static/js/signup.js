document.addEventListener("DOMContentLoaded", () => {
  const signupForm = document.querySelector("form");

  if (signupForm) {
    signupForm.addEventListener("submit", (e) => {
      const email = signupForm.querySelector("input[name='email']").value.trim();
      const password = signupForm.querySelector("input[name='password']").value.trim();

      const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

      if (!email || !password) {
        e.preventDefault();
        showToast("❗ Please fill out both email and password.");
        return;
      }

      if (!emailPattern.test(email)) {
        e.preventDefault();
        showToast("⚠️ Please enter a valid email address.");
        return;
      }

      if (password.length < 6) {
        e.preventDefault();
        showToast("🔒 Password must be at least 6 characters long.");
        return;
      }

      // Disable button on submit
      const button = signupForm.querySelector('button[type="submit"]');
      if (button) {
        button.disabled = true;
        button.innerHTML = '⏳ Creating account...';
      }
    });
  }

  function showToast(message) {
    const toast = document.createElement("div");
    toast.className =
      "fixed bottom-5 right-5 px-4 py-2 bg-red-500 text-white text-sm rounded shadow-lg z-50 animate-fade-in";
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("opacity-0");
      setTimeout(() => toast.remove(), 500);
    }, 2500);
  }
});
