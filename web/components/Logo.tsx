export default function Logo() {
  return (
    <span
      className="brand-inner"
      style={{ display: "inline-flex", alignItems: "center", gap: 10 }}
    >
      {/* Full logo lockup provided by the user; multiply blend drops its
          light background onto the page gray so it reads as bare ink. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="/logo.jpeg"
        alt="hoichoi"
        style={{
          height: 42,
          width: "auto",
          display: "block",
          mixBlendMode: "multiply",
        }}
      />
      <span className="brand-word">
        <span className="brand-x">x</span>
        <span className="brand-name">ReFrame</span>
      </span>
    </span>
  );
}
