<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcribe and Classify Customer Service Calls</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            text-align: center;
        }
        .container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            max-width: 600px;
            width: 100%;
        }
        h1 {
            color: #333;
            font-size: 24px;
            margin-bottom: 20px;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .message {
            margin-top: 20px;
            color: green;
        }
        .error {
            color: red;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Transcribe and Classify Customer Service Calls</h1>
        <form id="upload-form" enctype="multipart/form-data">
            <input type="file" name="mp3_files" id="mp3_files" accept=".mp3" multiple>
            <button type="submit">Upload and Process</button>
        </form>

        <div id="result-table"></div>
        <p class="message" id="success-message" style="display: none;">Transcripts and classifications have been added to the CSV file in the GitHub directory.</p>
        <p class="error" id="error-message" style="display: none;"></p>
    </div>

    <script>
        document.getElementById("upload-form").onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData();
            const mp3Files = document.getElementById("mp3_files").files;

            for (const file of mp3Files) {
                formData.append("mp3_files", file);
            }

            try {
                const response = await fetch("/upload", {
                    method: "POST",
                    body: formData
                });
                
                const data = await response.json();

                if (response.ok) {
                    // Generate a table to display transcripts and classifications
                    let tableHTML = `<table>
                                        <thead>
                                            <tr>
                                                <th>Call ID</th>
                                                <th>Transcript</th>
                                                <th>Classification</th>
                                            </tr>
                                        </thead>
                                        <tbody>`;
                    
                    data.results.forEach(row => {
                        tableHTML += `<tr>
                                        <td>${row.call_id}</td>
                                        <td>${row.transcript}</td>
                                        <td>${row.classification}</td>
                                      </tr>`;
                    });

                    tableHTML += `</tbody></table>`;
                    
                    document.getElementById("result-table").innerHTML = tableHTML;
                    document.getElementById("success-message").style.display = "block";
                    document.getElementById("error-message").style.display = "none";
                } else {
                    throw new Error(data.detail || "An error occurred while processing the files.");
                }
            } catch (error) {
                document.getElementById("success-message").style.display = "none";
                document.getElementById("error-message").textContent = error.message;
                document.getElementById("error-message").style.display = "block";
            }
        };
    </script>
</body>
</html>
