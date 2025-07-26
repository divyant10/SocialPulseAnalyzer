document.addEventListener("DOMContentLoaded", () => {

  // ✅ VIRALITY SCORE GAUGE
  if (typeof Plotly !== "undefined") {
    const viralityChartDiv = document.getElementById("viralityChart");

    if (viralityChartDiv) {
      const score = typeof viralityScore !== 'undefined' ? parseFloat(viralityScore) : 0;

      const data = [{
        type: "indicator",
        mode: "gauge+number",
        value: score,
        title: { text: "Overall", font: { size: 20 } },
        gauge: {
          shape: "semi",
          axis: {
            range: [0, 100],
            tickwidth: 1,
            tickcolor: "#999",
            // You can optionally make tick labels smaller if they're still crowding
            // tickfont: { size: 10 } // Uncomment and adjust if needed
          },
          bar: { color: "#f43f5e", thickness: 0.1 },
          bgcolor: "transparent",
          borderwidth: 1,
          bordercolor: "#ccc",
          steps: [
            { range: [0, 30], color: "#fecaca" },
            { range: [30, 70], color: "#fef08a" },
            { range: [70, 100], color: "#bbf7d0" }
          ],
        }
      }];

      const layout = {
        // --- MORE AGGRESSIVE MARGINS & SLIGHTLY LARGER CHART ---
        width: 380, // Increased width
        height: 300, // Increased height significantly for vertical labels
        margin: { t: 60, b: 60, l: 60, r: 60 }, // Increased margins significantly
        // --- END OF UPDATED SECTION ---
        paper_bgcolor: "transparent",
        font: { color: "#1f2937", family: "Arial" }
      };

      Plotly.newPlot("viralityChart", data, layout, { responsive: true });
    } else {
      console.warn("Element with ID 'viralityChart' not found. Cannot render Virality Gauge.");
    }
  } else {
    console.error("Plotly library not loaded. Cannot render charts.");
  }


  // ✅ SENTIMENT BAR CHART
  const sentimentChartDiv = document.getElementById("sentimentChart");

  if (sentimentChartDiv && typeof Plotly !== "undefined") {
    if (typeof sentimentScores !== 'undefined' && sentimentScores.length > 0 &&
        typeof sentimentLabels !== 'undefined' && sentimentLabels.length > 0) {
      try {
        Plotly.newPlot("sentimentChart", [{
          x: sentimentLabels,
          y: sentimentScores,
          type: 'bar',
          marker: {
            color: ['#22c55e', '#facc15', '#ef4444']
          }
        }], {
          title: 'Sentiment Breakdown',
          xaxis: { title: 'Sentiment' },
          yaxis: { title: 'Score' },
          margin: { t: 40, b: 40, l: 40, r: 40 }, // Keep these margins as they seemed okay
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#1f2937", family: "Arial" }
        }, { responsive: true });
      } catch (e) {
        console.error("Failed to plot sentiment data:", e);
      }
    } else {
      sentimentChartDiv.innerHTML = '<p class="text-center text-gray-500">No sentiment data available.</p>';
    }
  } else if (!sentimentChartDiv) {
      console.warn("Element with ID 'sentimentChart' not found. Cannot render Sentiment Chart.");
  } else {
      console.error("Plotly library not loaded. Cannot render charts.");
  }


  // ✅ AUTO REDIRECT AFTER 30 SECONDS FROM SUMMARY PAGE
  if (window.location.pathname === "/summary") {
    setTimeout(() => {
      window.location.href = "/dashboard";
    }, 30000);
  }


  // ✅ PLATFORM LOGO AND CARD GLOW
  const platformTextElement = document.getElementById("platformText");
  const platformIcon = document.getElementById("platformIcon");
  const platformGlow = document.getElementById("platformCard");

  if (platformTextElement && platformGlow) {
    const platform = platformTextElement.textContent?.trim().toLowerCase();
    let logo = "🌐",
      glowClass = "shadow-gray-400";

    switch (platform) {
      case "instagram":
        logo = "📸";
        glowClass = "shadow-pink-500";
        break;
      case "twitter":
        logo = "🐦";
        glowClass = "shadow-blue-400";
        break;
      case "youtube":
        logo = "▶️";
        glowClass = "shadow-red-500";
        break;
      case "facebook":
        logo = "📘";
        glowClass = "shadow-blue-600";
        break;
      default:
        break;
    }

    if (platformIcon) platformIcon.textContent = logo;
    platformGlow.classList.add(glowClass, "shadow-lg");
  }


  // ✅ DELETE HISTORY ITEM + TOAST
  document.querySelectorAll(".delete-history").forEach((icon) => {
    icon.addEventListener("click", async (event) => {
      event.stopPropagation();
      const timestamp = icon.getAttribute("data-timestamp");

      const res = await fetch("/delete_history_item", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ timestamp })
      });

      const result = await res.json();
      if (result.success) {
        icon.closest(".history-item")?.remove();
        showToast("Deleted analysis history item.");
      } else {
        showToast("Failed to delete history item.", true);
      }
    });
  });


  // ✅ CLICKABLE HISTORY ITEM — REFILL FORM AND RE-ANALYZE
  document.querySelectorAll(".history-item.clickable").forEach((card) => {
    card.addEventListener("click", () => {
      const caption = card.getAttribute("data-caption");
      const platform = card.getAttribute("data-platform");
      const likes = card.getAttribute("data-likes");
      const views = card.getAttribute("data-views");
      const hashtags = card.getAttribute("data-hashtags");

      localStorage.setItem("prefill_data", JSON.stringify({
        caption,
        platform,
        likes,
        views,
        hashtags
      }));
      window.location.href = "/main";
    });
  });


  // ✅ SMOOTH SCROLL TO VIRALITY SCORE
  const viralityChartContainer = document.getElementById("viralityChart");
  if (viralityChartContainer) {
    setTimeout(() => {
      viralityChartContainer.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 500);
  }


  // ✅ TOAST MESSAGE FUNCTION
  function showToast(message, isError = false) {
    const toast = document.createElement("div");
    toast.className = `fixed bottom-5 right-5 px-4 py-2 rounded-lg shadow-md text-white text-sm transition-all duration-300 z-50 ${
      isError ? "bg-red-500" : "bg-green-500"
    }`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("opacity-0");
      setTimeout(() => toast.remove(), 500);
    }, 2000);
  }
});