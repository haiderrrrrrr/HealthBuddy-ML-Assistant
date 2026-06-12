// Select all accordion-content elements (regardless of their group)
document.querySelectorAll(".accordion-content").forEach((item) => {
  item.querySelector("header").addEventListener("click", () => {
    const isOpen = item.classList.contains("open");

    document.querySelectorAll(".accordion-content").forEach((content) => {
      content.classList.remove("open");
      content.querySelector(".description").style.height = "0";
      content.querySelector("i").classList.replace("fa-minus", "fa-plus");
    });

    if (!isOpen) {
      item.classList.add("open");
      const description = item.querySelector(".description");
      description.style.height = `${description.scrollHeight + 15}px`;
      item.querySelector("i").classList.replace("fa-plus", "fa-minus");
    }
  });
});

