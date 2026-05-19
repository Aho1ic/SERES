from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/test", methods=["POST"])
def test():
	response = {
		"code": 0,
		"data": {
			"boxId": "20",
			"task_id": "1960216899426193408",
			"fileUrl": "http://218.92.176.50:9000/tenant-0/DJI_20250714144200_0009_T.jpeg",
			"droneSn": "1581F8HGX253E00A04A7",
			"thirdGroupId": "36180",
			'takeType': 1
		}
	}
	return jsonify(response)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=58084, debug=True)
