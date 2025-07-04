document.addEventListener("DOMContentLoaded", () => {
    // Create floating coins for the main page
    if (document.getElementById("floatingCoins")) {
      createFloatingCoins()
    }
  
    // Add hover effects to module cards
    const moduleCards = document.querySelectorAll(".module-card")
    moduleCards.forEach((card) => {
      card.addEventListener("mouseenter", function () {
        const thoughtBubble = this.querySelector(".thought-bubble")
        if (thoughtBubble) {
          thoughtBubble.classList.remove("hidden")
        }
      })
  
      card.addEventListener("mouseleave", function () {
        const thoughtBubble = this.querySelector(".thought-bubble")
        if (thoughtBubble) {
          thoughtBubble.classList.add("hidden")
        }
      })
    })
  
    // Initialize Smart Spenders page functionality
    if (document.querySelector(".smart-spenders-page")) {
      initSmartSpendersPage()
    }
  
    // Initialize Teen Investors page functionality
    if (document.querySelector(".teen-investors-page")) {
      initTeenInvestorsPage()
    }
  })
  
  // Create floating coins animation
  function createFloatingCoins() {
    const floatingCoinsContainer = document.getElementById("floatingCoins")
  
    for (let i = 0; i < 15; i++) {
      const coin = document.createElement("div")
      coin.className = "coin"
      coin.textContent = "$"
  
      // Random positioning
      const randomX = Math.random() * 100
      const randomDelay = Math.random() * 10
      const randomDuration = 15 + Math.random() * 20
  
      coin.style.left = `${randomX}%`
      coin.style.animationDelay = `${randomDelay}s`
      coin.style.animationDuration = `${randomDuration}s`
  
      floatingCoinsContainer.appendChild(coin)
    }
  }
  
  // Initialize Smart Spenders page functionality
  function initSmartSpendersPage() {
    // Dreamboard functionality
    const goalItems = document.querySelectorAll(".goal-item")
    const savingsSlider = document.getElementById("savingsSlider")
    const weeklyAmount = document.getElementById("weeklyAmount")
    const selectedGoal = document.getElementById("selectedGoal")
    const weeksToGoal = document.getElementById("weeksToGoal")
  
    if (goalItems.length && savingsSlider) {
      // Set default selected goal
      goalItems[0].classList.add("selected")
  
      // Goal selection
      goalItems.forEach((item) => {
        item.addEventListener("click", function () {
          goalItems.forEach((g) => g.classList.remove("selected"))
          this.classList.add("selected")
          updateSavingsCalculation()
        })
      })
  
      // Savings slider
      savingsSlider.addEventListener("input", function () {
        weeklyAmount.textContent = this.value
        document.getElementById("savingsAmount").textContent = `$${this.value}`
        updateSavingsCalculation()
      })
  
      // Initial calculation
      updateSavingsCalculation()
    }
  
    // Spin wheel segments
    const spinWheel = document.getElementById("spinWheel")
    if (spinWheel) {
      const colors = ["#FF9AA2", "#FFB7B2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA"]
  
      colors.forEach((color, index) => {
        const segment = document.createElement("div")
        segment.className = "spin-wheel-segment"
        segment.style.backgroundColor = color
        segment.style.transform = `rotate(${index * 60}deg)`
        spinWheel.appendChild(segment)
      })
    }
  }
  
  // Calculate savings time for Smart Spenders page
  function updateSavingsCalculation() {
    const savingsAmount = Number.parseInt(document.getElementById("savingsSlider").value)
    const selectedGoalItem = document.querySelector(".goal-item.selected")
  
    if (selectedGoalItem && savingsAmount) {
      const goalName = selectedGoalItem.querySelector(".goal-name").textContent
      const goalPrice = Number.parseInt(selectedGoalItem.querySelector(".goal-price").textContent.replace("$", ""))
  
      document.getElementById("selectedGoal").textContent = goalName
      const weeks = Math.ceil(goalPrice / savingsAmount)
      document.getElementById("weeksToGoal").textContent = weeks
    }
  }
  
  // Initialize Teen Investors page functionality
  function initTeenInvestorsPage() {
    // Investment slider functionality
    const investmentSlider = document.getElementById("investmentSlider")
  
    if (investmentSlider) {
      investmentSlider.addEventListener("input", function () {
        const investmentRatio = Number.parseInt(this.value)
        const savingsRatio = 100 - investmentRatio
  
        // Update percentages
        document.getElementById("investmentPercentage").textContent = `${investmentRatio}%`
        document.getElementById("savingsPercentage").textContent = `${savingsRatio}%`
  
        // Calculate results after 10 years
        const initialAmount = 1000
        const savingsResult = Math.round((initialAmount * Math.pow(1.03, 10) * savingsRatio) / 100)
        const investmentResult = Math.round((initialAmount * Math.pow(1.08, 10) * investmentRatio) / 100)
  
        // Update results
        document.getElementById("savingsResult").textContent = savingsResult
        document.getElementById("investmentResult").textContent = investmentResult
      })
  
      // Trigger initial calculation
      investmentSlider.dispatchEvent(new Event("input"))
    }
  
    // Add animation to stock prices
    setInterval(() => {
      const stockItems = document.querySelectorAll(".stock-item")
  
      stockItems.forEach((item) => {
        const priceElement = item.querySelector(".price-value")
        const changeElement = item.querySelector(".price-change")
  
        if (priceElement && changeElement) {
          // Get current price
          const currentPrice = Number.parseFloat(priceElement.textContent.replace("$", ""))
  
          // Random price change between -5% and +5%
          const changePercent = (Math.random() * 10 - 5) / 100
          const newPrice = Math.max(1, currentPrice * (1 + changePercent))
  
          // Update price
          priceElement.textContent = `$${newPrice.toFixed(2)}`
  
          // Update change indicator
          const changeValue = Math.abs((changePercent * 100).toFixed(1))
          const isPositive = changePercent >= 0
  
          changeElement.className = `price-change ${isPositive ? "positive" : "negative"}`
          changeElement.innerHTML = `
            <svg class="trend-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="${isPositive ? "23 6 13.5 15.5 8.5 10.5 1 18" : "23 18 13.5 8.5 8.5 13.5 1 6"}"></polyline>
              <polyline points="${isPositive ? "17 6 23 6 23 12" : "17 18 23 18 23 12"}"></polyline>
            </svg>
            ${changeValue}%
          `
        }
      })
    }, 5000) // Update every 5 seconds
  }
  
  // Add comic panel animation effects
  document.querySelectorAll(".comic-panel").forEach((panel) => {
    panel.addEventListener("mouseenter", function () {
      this.style.transform = "scale(1.05) rotate(1deg)"
    })
  
    panel.addEventListener("mouseleave", function () {
      this.style.transform = ""
    })
  })
  
  // Add comic button effects
  document.querySelectorAll(".comic-button").forEach((button) => {
    button.addEventListener("mouseenter", function () {
      this.style.transform = "scale(1.05)"
    })
  
    button.addEventListener("mouseleave", function () {
      this.style.transform = ""
    })
  
    button.addEventListener("mousedown", function () {
      this.style.transform = "translateX(2px) translateY(2px)"
      this.style.boxShadow = "2px 2px 0px 0px rgba(0,0,0,0.8)"
    })
  
    button.addEventListener("mouseup", function () {
      this.style.transform = "scale(1.05)"
      this.style.boxShadow = "4px 4px 0px 0px rgba(0,0,0,0.8)"
    })
  })
  