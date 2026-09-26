"use client";

import { useState } from "react";

const IMAGE_EXT = /\.(jpe?g|png|webp|gif|avif)$/i;
const VIDEO_EXT = /\.(mp4|webm|mov|m4v)$/i;

export function isImageKey(key: string): boolean {
  return IMAGE_EXT.test(key);
}

export function previewUrl(key: string): string {
  return `/api/serve?key=${encodeURIComponent(key)}`;
}

/**
 * Thumbnail for an `outputs.preview_key`.
 *
 * preview_key is meant to be an image, but a video key would render as a
 * broken-image icon, so the element is picked from the key's extension and a
 * failed load degrades to a labelled box — the browser's broken-image glyph
 * never appears. Video keys render muted and metadata-only, so a legacy row
 * still previews something cheap.
 */
export function PreviewThumb({
  previewKey,
  alt,
  className,
  style,
  emptyLabel = "No preview",
}: {
  previewKey: string | null;
  alt: string;
  className?: string;
  style?: React.CSSProperties;
  emptyLabel?: string;
}) {
  const [broken, setBroken] = useState(false);

  if (!previewKey || broken) {
    return (
      <div
        className={className ? `thumb-fallback ${className}` : "thumb-fallback"}
        style={style}
        role="img"
        aria-label={alt}
      >
        <span className="micro">{emptyLabel}</span>
      </div>
    );
  }

  const src = previewUrl(previewKey);

  if (VIDEO_EXT.test(previewKey)) {
    return (
      <video
        className={className}
        style={style}
        src={src}
        aria-label={alt}
        muted
        playsInline
        preload="metadata"
        onError={() => setBroken(true)}
      />
    );
  }

  return (
    /* eslint-disable-next-line @next/next/no-img-element */
    <img
      className={className}
      style={style}
      src={src}
      alt={alt}
      onError={() => setBroken(true)}
    />
  );
}
