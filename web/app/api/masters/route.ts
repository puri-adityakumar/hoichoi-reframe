import { NextResponse } from "next/server";
import { listMasters } from "@/db";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const masters = await listMasters();
    return NextResponse.json({ masters });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "DB error", masters: [] },
      { status: 200 }
    );
  }
}
