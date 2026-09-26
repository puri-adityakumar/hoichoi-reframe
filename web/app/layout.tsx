import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "hoichoi-reframe — P4 Creative Reformatting Engine",
  description:
    "Upload one master, get every platform version with the subject still in frame.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav>
          <Link className="brand" href="/">
            hoichoi<span>-</span>reframe
          </Link>
          <Link className="navlink" href="/">
            Upload
          </Link>
          <Link className="navlink" href="/library">
            Library
          </Link>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}
