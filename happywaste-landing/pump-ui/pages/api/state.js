import { execFile } from "child_process";
import path from "path";

export default function handler(req, res) {
  const pythonBin = process.env.PYTHON_BIN || "python";
  const scriptPath = path.join(process.cwd(), "..", "backend", "snapshot.py");

  execFile(pythonBin, [scriptPath], { timeout: 15000 }, (error, stdout, stderr) => {
    if (error) {
      console.error("snapshot.py error", error, stderr);
      res.status(500).json({ error: "snapshot_failed" });
      return;
    }

    try {
      const data = JSON.parse(stdout.toString());
      res.status(200).json(data);
    } catch (e) {
      console.error("snapshot.py invalid JSON", e, stdout.toString());
      res.status(500).json({ error: "invalid_json" });
    }
  });
}
