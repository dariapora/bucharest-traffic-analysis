import sqlite3
import json
from flask import Flask, jsonify, request, send_from_directory
from pathlib import Path

app = Flask(__name__, static_folder="static")
DB_PATH = Path(__file__).parent / "traffic.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/api/dates")
def api_dates():
    conn = get_db()
    rows = conn.execute(
        "SELECT date, day_of_week, day_of_week_num, is_weekend FROM dates ORDER BY date"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/segments")
def api_segments():
    conn = get_db()
    rows = conn.execute(
        "SELECT segment_id, street_name, speed_limit, frc, distance, latitude, longitude, geometry FROM segments"
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("geometry"):
            d["geometry"] = json.loads(d["geometry"])
        result.append(d)
    return jsonify(result)


@app.route("/api/streets")
def api_streets():
    conn = get_db()
    rows = conn.execute(
        "SELECT DISTINCT street_name FROM segments WHERE street_name IS NOT NULL ORDER BY street_name"
    ).fetchall()
    conn.close()
    return jsonify([r["street_name"] for r in rows])


@app.route("/api/traffic")
def api_traffic():
    date = request.args.get("date")
    time = request.args.get("time", type=float)
    streets = [s for s in request.args.getlist("street") if s]
    if not streets:
        street = request.args.get("street")
        if street:
            streets = [street]

    if not date or time is None:
        return jsonify({"error": "date and time required"}), 400

    conn = get_db()

    if streets:
        placeholders = ", ".join(["?"] * len(streets))
        query = f"""
            SELECT t.segment_id, t.time_numeric, t.traffic_state,
                   t.speed_ratio, t.median_speed_val, t.total_samples,
                   s.latitude, s.longitude, s.street_name, s.speed_limit
            FROM traffic t
            JOIN segments s ON t.segment_id = s.segment_id
            WHERE t.date = ? AND t.time_numeric = ? AND s.street_name IN ({placeholders})
        """
        rows = conn.execute(query, (date, time, *streets)).fetchall()
    else:
        rows = conn.execute("""
            SELECT t.segment_id, t.time_numeric, t.traffic_state,
                   t.speed_ratio, t.median_speed_val, t.total_samples,
                   s.latitude, s.longitude, s.street_name, s.speed_limit
            FROM traffic t
            JOIN segments s ON t.segment_id = s.segment_id
            WHERE t.date = ? AND t.time_numeric = ?
        """, (date, time)).fetchall()

    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/traffic/day")
def api_traffic_day():
    date = request.args.get("date")
    streets = [s for s in request.args.getlist("street") if s]
    if not streets:
        street = request.args.get("street")
        if street:
            streets = [street]

    if not date:
        return jsonify({"error": "date required"}), 400

    conn = get_db()

    if streets:
        placeholders = ", ".join(["?"] * len(streets))
        query = f"""
            SELECT t.segment_id, t.time_numeric, t.traffic_state,
                   t.speed_ratio, t.median_speed_val, t.total_samples,
                   s.latitude, s.longitude, s.street_name, s.speed_limit
            FROM traffic t
            JOIN segments s ON t.segment_id = s.segment_id
            WHERE t.date = ? AND s.street_name IN ({placeholders})
            ORDER BY t.time_numeric
        """
        rows = conn.execute(query, (date, *streets)).fetchall()
    else:
        rows = conn.execute("""
            SELECT t.segment_id, t.time_numeric, t.traffic_state,
                   t.speed_ratio, t.median_speed_val, t.total_samples,
                   s.latitude, s.longitude, s.street_name, s.speed_limit
            FROM traffic t
            JOIN segments s ON t.segment_id = s.segment_id
            WHERE t.date = ?
            ORDER BY t.time_numeric
        """, (date,)).fetchall()

    conn.close()

    by_time = {}
    for r in rows:
        d = dict(r)
        t = d["time_numeric"]
        if t not in by_time:
            by_time[t] = []
        by_time[t].append(d)

    return jsonify(by_time)


@app.route("/api/traffic/segment/<segment_id>")
def api_segment_day(segment_id):
    date = request.args.get("date")
    if not date:
        return jsonify({"error": "date required"}), 400

    conn = get_db()
    rows = conn.execute("""
        SELECT t.time_numeric, t.traffic_state,
               t.speed_ratio, t.median_speed_val, t.total_samples,
               s.street_name, s.speed_limit, s.latitude, s.longitude
        FROM traffic t
        JOIN segments s ON t.segment_id = s.segment_id
        WHERE t.segment_id = ? AND t.date = ?
        ORDER BY t.time_numeric
    """, (segment_id, date)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/overview")
def api_overview():
    date = request.args.get("date")
    streets = [s for s in request.args.getlist("street") if s]
    if not streets:
        street = request.args.get("street")
        if street:
            streets = [street]

    if not date:
        return jsonify({"error": "date required"}), 400

    conn = get_db()

    if streets:
        placeholders = ", ".join(["?"] * len(streets))
        query = f"""
            SELECT t.time_numeric,
                   SUM(CASE WHEN t.traffic_state = 'free_flow' THEN 1 ELSE 0 END) as cnt_free_flow,
                   SUM(CASE WHEN t.traffic_state = 'slow' THEN 1 ELSE 0 END) as cnt_slow,
                   SUM(CASE WHEN t.traffic_state = 'congested' THEN 1 ELSE 0 END) as cnt_congested,
                   COUNT(*) as total,
                   AVG(t.speed_ratio) as avg_speed_ratio,
                   SUM(t.total_samples) as total_samples
            FROM traffic t
            JOIN segments s ON t.segment_id = s.segment_id
            WHERE t.date = ? AND s.street_name IN ({placeholders})
            GROUP BY t.time_numeric
            ORDER BY t.time_numeric
        """
        rows = conn.execute(query, (date, *streets)).fetchall()
    else:
        rows = conn.execute("""
            SELECT time_numeric,
                   SUM(CASE WHEN traffic_state = 'free_flow' THEN 1 ELSE 0 END) as cnt_free_flow,
                   SUM(CASE WHEN traffic_state = 'slow' THEN 1 ELSE 0 END) as cnt_slow,
                   SUM(CASE WHEN traffic_state = 'congested' THEN 1 ELSE 0 END) as cnt_congested,
                   COUNT(*) as total,
                   AVG(speed_ratio) as avg_speed_ratio,
                   SUM(total_samples) as total_samples
            FROM traffic
            WHERE date = ?
            GROUP BY time_numeric
            ORDER BY time_numeric
        """, (date,)).fetchall()

    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        total = d["total"] or 1
        d["avg_free_flow"] = round(d["cnt_free_flow"] / total * 100, 1)
        d["avg_slow"] = round(d["cnt_slow"] / total * 100, 1)
        d["avg_congested"] = round(d["cnt_congested"] / total * 100, 1)
        result.append(d)

    return jsonify(result)


@app.route("/api/stats")
def api_stats():
    conn = get_db()
    seg_count = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    traffic_count = conn.execute("SELECT COUNT(*) FROM traffic").fetchone()[0]
    date_count = conn.execute("SELECT COUNT(*) FROM dates").fetchone()[0]
    conn.close()
    return jsonify({
        "segments": seg_count,
        "traffic_rows": traffic_count,
        "dates": date_count
    })


if __name__ == "__main__":
    if not DB_PATH.exists():
        print("ERROR: traffic.db not found. Restore the database before starting the portal.")
        exit(1)
    print("Starting Traffic Portal on http://localhost:5000")
    app.run(debug=True, port=5000)