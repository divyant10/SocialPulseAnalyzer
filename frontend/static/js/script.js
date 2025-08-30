document.addEventListener('DOMContentLoaded', () => {
  const form = document.querySelector('form');
  const youtubeFields = document.getElementById('youtube-extra-fields');

  if (form) {
    form.classList.add('animate-fade-in');
  }

  const firstInput = form?.querySelector('select, input, textarea');
  if (firstInput) {
    firstInput.focus();
  }

  const prefillData = JSON.parse(localStorage.getItem("prefill_data"));
  if (prefillData && form) {
    form.querySelector('textarea[name="caption"]').value = prefillData.caption || "";
    form.querySelector('input[name="likes"]').value = prefillData.likes || "";
    form.querySelector('input[name="views"]').value = prefillData.views || "";
    form.querySelector('input[name="hashtags"]').value = prefillData.hashtags || "";

    const dropdownBtn = document.querySelector('[x-data] button');
    if (dropdownBtn && prefillData.platform) {
      dropdownBtn.click(); 
      setTimeout(() => {
        const options = document.querySelectorAll('[x-data] ul li');
        options.forEach(opt => {
          if (opt.textContent.trim().toLowerCase() === prefillData.platform.toLowerCase()) {
            opt.click();
          }
        });
      }, 100);
    }

    localStorage.removeItem("prefill_data"); 
  }

  if (form) {
    form.addEventListener('submit', (e) => {
      const likesInput = form.querySelector('input[name="likes"]');
      const viewsInput = form.querySelector('input[name="views"]');

      const shorthandRegex = /^\d+(\.\d+)?[KMB]?$/i;

      const isLikesValid = shorthandRegex.test(likesInput.value.trim());
      const isViewsValid = shorthandRegex.test(viewsInput.value.trim());

      if (!isLikesValid || !isViewsValid) {
        e.preventDefault();
        showToast("❌ Please enter valid Likes/Views like 10K, 2.3M or 100000.");
        return;
      }

      const button = form.querySelector('button[type="submit"]');
      if (button) {
        button.disabled = true;
        button.innerHTML = '⏳ Analyzing...';
      }
    });
  }

  if (window.location.href.includes("analyze")) {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }


  const observer = new MutationObserver(() => {
    const selectedPlatformEl = document.querySelector('[x-text="selected.name"]');
    const selectedPlatform = selectedPlatformEl ? selectedPlatformEl.textContent.trim().toLowerCase() : "";

    if (selectedPlatform === 'youtube') {
      youtubeFields?.classList.remove('hidden');
    } else {
      youtubeFields?.classList.add('hidden');
    }
  });

  const dropdown = document.querySelector('[x-data]');
  if (dropdown) {
    observer.observe(dropdown, {
      subtree: true,
      childList: true,
      characterData: true
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
