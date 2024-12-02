---
layout: default  
title: graph-natural-language-query 
---

# Graph visualization based on natural language/LLM interface

<a href="https://github.com/PrabuAppDev/graph-query-REST/blob/main/README.md" target="_blank">View the latest Graph visualization project on GitHub</a>

# Natural language based graph generator

## Key highlights
- Successfully integrated a Flask-based Python server to process system interaction queries and generate JSON responses.
- Developed a D3.js-powered HTML page to visualize the interactions as a graph, which dynamically fetches data from the Flask server.
- Implemented vector database functionality using Qdrant to enable efficient retrieval of system interaction context.
- Utilized OpenAI's API for natural language understanding to process queries and map them to system interaction data.
- Verified functionality with a sample query, as shown below, which demonstrates the system graph visualization for the query "What are the systems that interact with Admissions Portal?"

![System Interaction Visualization](/assets/images/D3-HTML-screen-print.gif)

---

## Credits
This project leverages concepts learned from:  

- My Master's in Computer Science program at Georgia Tech, particularly the Data and Visual Analytics course: [D3.js tutorial from CSE 6242: Data and Visual Analytics](https://cdnapisec.kaltura.com/html5/html5lib/v2.82.1/mwEmbedFrame.php/p/2019031/uiconf_id/40436601?wid=1_2jwnn9ky&iframeembed=true&playerId=kaltura_player_5b686240b32ef&flashvars%5BplaylistAPI.kpl0Id%5D=1_16jro99t&flashvars%5BplaylistAPI.autoContinue%5D=true&flashvars%5BplaylistAPI.autoInsert%5D=true&flashvars%5Bks%5D=&flashvars%5BlocalizationCode%5D=en&flashvars%5BimageDefaultDuration%5D=30&flashvars%5BleadWithHTML5%5D=true&flashvars%5BforceMobileHTML5%5D=true&flashvars%5BnextPrevBtn.plugin%5D=true&flashvars%5BsideBarContainer.plugin%5D=true&flashvars%5BsideBarContainer.position%5D=left&flashvars%5BsideBarContainer.clickToClose%5D=true&flashvars%5Bchapters.plugin%5D=true&flashvars%5Bchapters.layout%5D=vertical&flashvars%5Bchapters.thumbnailRotator%5D=false&flashvars%5BstreamSelector.plugin%5D=true&flashvars%5BEmbedPlayer.SpinnerTarget%5D=videoHolder&flashvars%5BdualScreen.plugin%5D=true){:target="_blank"} for helping me learn graph visualization techniques  
- The Gang of Four (GoF) design patterns, which helped me appreciate the role of design patterns in system integrations.
- The Sun Certified Enterprise Architect (SCEA) curriculum, which served as a valuable reference for integration architecture. 

--- 