import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RallyLens — Badminton Stroke & Rally Analytics",
  description:
    "Upload a badminton shot clip — RallyLens identifies the stroke, explains the decision with Grad-CAM, forecasts the next shot, and coaches the response.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
