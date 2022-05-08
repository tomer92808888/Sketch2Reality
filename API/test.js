const request = require("request");
const fs = require("fs");

binary_input =fs.createReadStream('sample-input.png');


var formData = {
  file: binary_input,
};

request.post(
  {
    headers: {
      "Content-Type": "multipart/form-data",
      connection: "keep-alive",
    },
    url: "http://localhost:8000/predict",
    formData: formData,
  },
  function (err, res, body) {
    console.log(body);
    fs.writeFileSync("sample-output.png", Buffer.from(body), (err) => {
      if (err) throw err;
      console.log("File saved!");
    });
  }
);