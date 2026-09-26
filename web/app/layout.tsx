import type { Metadata } from "next";
import Link from "next/link";
import Logo from "../components/Logo";
import "./globals.css";

const FAVICON =
  "data:image/svg+xml," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#16150f" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8"/><path d="M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8"/><path d="M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16"/><path d="M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16"/><rect x="9" y="9" width="6" height="6" stroke-width="1.5"/></svg>'
  );

export const metadata: Metadata = {
  title: "x ReFrame — creative reformatting engine",
  description:
    "Upload one master, get every platform version with the subject still in frame.",
  icons: {
    icon: [{ url: FAVICON, type: "image/svg+xml" }],
  },
};

function DashboardIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <rect x="4" y="4" width="6.5" height="6.5" rx="1" />
      <rect x="13.5" y="4" width="6.5" height="6.5" rx="1" />
      <rect x="4" y="13.5" width="6.5" height="6.5" rx="1" />
      <rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1" />
    </svg>
  );
}

function LibraryIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3.5 7.5A1.5 1.5 0 0 1 5 6h4.6a1.5 1.5 0 0 1 1.2.6l1 1.4h7.2a1.5 1.5 0 0 1 1.5 1.5v8a1.5 1.5 0 0 1-1.5 1.5H5a1.5 1.5 0 0 1-1.5-1.5v-10z" />
    </svg>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav>
          <Link className="brand" href="/">
            <Logo />
          </Link>
          <div className="nav-center">
            <Link className="navlink" href="/">
              <DashboardIcon />
              Dashboard
            </Link>
            <Link className="navlink" href="/library">
              <LibraryIcon />
              Library
            </Link>
          </div>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}
