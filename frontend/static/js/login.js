document.addEventListener("DOMContentLoaded", () => {
  const loginForm = document.querySelector("form");
  const passwordInput = document.querySelector("input[name='password']");
  const toggleIcon = document.getElementById("togglePassword");

  if (loginForm) {
    loginForm.addEventListener("submit", (e) => {
      const emailInput = loginForm.querySelector("input[name='email']");
      const email = emailInput.value.trim();
      const password = passwordInput.value.trim();

      if (!email || !password) {
        e.preventDefault();
        showToast("⚠️ Please enter both email and password.", true);
        return;
      }

      if (!validateEmail(email)) {
        e.preventDefault();
        showToast("⚠️ Please enter a valid email address.", true);
        return;
      }

      if (password.length < 6) {
        e.preventDefault();
        showToast("⚠️ Password must be at least 6 characters.", true);
        return;
      }
    });
  }

  if (toggleIcon && passwordInput) {
    toggleIcon.addEventListener("click", () => {
      const isPassword = passwordInput.type === "password";
      passwordInput.type = isPassword ? "text" : "password";
      toggleIcon.classList.toggle("text-blue-500");
      toggleIcon.textContent = isPassword ? "🙈" : "👁️";
    });
  }

  function validateEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }

  function showToast(message, isError = false) {
    const toast = document.createElement("div");
    toast.className = `fixed bottom-5 right-5 px-4 py-2 rounded-lg shadow-md text-white text-sm transition-all duration-300 ${
      isError ? "bg-red-500" : "bg-green-500"
    } z-50`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("opacity-0");
      setTimeout(() => toast.remove(), 500);
    }, 2000);
  }
});
