// Variable to store the download key received from the server
let currentDownloadKey = null;
let currentFileName = null;
 
async function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  const output = document.getElementById("output");
  const downloadBtn = document.getElementById("downloadBtn");
 
  if (!fileInput.files.length) {
    alert("Please select a file first.");
    return;
  }
 
  // Reset state
  output.textContent = "Analyzing data...";
  downloadBtn.style.display = 'none';
 
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
 
  try {
    const response = await fetch("https://veridata-backend.onrender.com/upload-dataset/", {
      method: "POST",
      body: formData,
    });
    
    const data = await response.json();
 
    if (data.error) {
        output.textContent = "Analysis Error: " + data.error;
        return;
    }
 
    // Save key and filename for PDF download
    currentDownloadKey = data.download_key;
    currentFileName = data.filename;
    
    // Display the JSON output
    output.textContent = JSON.stringify(data.nutrition_label, null, 2);
    
    // Show the download button
    downloadBtn.style.display = 'block';
 
  } catch (error) {
    output.textContent = "Error communicating with API: " + error;
  }
}
 
async function downloadLabel() {
    if (!currentDownloadKey || !currentFileName) {
        alert("Please upload and analyze a file first.");
        return;
    }
 
    const downloadUrl = `https://veridata-backend.onrender.com/download-label/?key=${currentDownloadKey}&filename=${currentFileName}`;
 
    try {
        const response = await fetch(downloadUrl);
        
        if (!response.ok) {
             throw new Error(`HTTP error! status: ${response.status}`);
        }
 
        // Get the PDF content as a Blob
        const blob = await response.blob();
 
        // Create a temporary URL and link element to trigger the download
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `${currentFileName}_Nutrition_Label.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        
    } catch (error) {
        console.error("PDF Download Error:", error);
        alert("Failed to download PDF. The analysis cache may have expired or an error occurred on the server.");
    }
}