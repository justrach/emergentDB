import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "EmergentDB | Self-Optimizing Vector Database",
  description: "A self-optimizing vector database using MAP-Elites evolutionary algorithm. 44-193x faster than LanceDB, 10-25x faster than ChromaDB.",
  keywords: ["vector database", "embeddings", "HNSW", "MAP-Elites", "AI", "machine learning"],
  authors: [{ name: "EmergentDB" }],
  openGraph: {
    title: "EmergentDB | Self-Optimizing Vector Database",
    description: "MAP-Elites powered vector search. Automatically optimizes for your workload.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#09090B] text-white`}
      >
        {children}
      </body>
    </html>
  );
}
