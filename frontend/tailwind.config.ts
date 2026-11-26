import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Plus Jakarta Sans", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        page: "#f4f7f5",
        surface: "#ffffff",
        line: "#e5eae7",
        ink: "#0f172a",
        sub: "#64748b",
        court: {
          DEFAULT: "#0c3b2e",
          deep: "#092e24",
        },
        brand: {
          DEFAULT: "#10b981",
          dark: "#0a8f63",
          soft: "#e8f7f0",
        },
      },
      boxShadow: {
        card: "0 1px 2px rgba(16,24,40,0.04), 0 8px 24px rgba(16,24,40,0.06)",
        lift: "0 18px 44px rgba(9,46,36,0.12)",
        pop: "0 10px 26px rgba(16,185,129,0.32)",
      },
      backgroundImage: {
        hero: "radial-gradient(900px 420px at 88% -20%, rgba(16,185,129,0.28), transparent 60%), linear-gradient(135deg, #0c3b2e 0%, #092e24 100%)",
      },
    },
  },
  plugins: [],
};

export default config;
