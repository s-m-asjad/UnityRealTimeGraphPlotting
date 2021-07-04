# UnityRealTimeGraphPlotting
This Unity project contains the scripts &amp; details of how to create a graph that can be updated in real-time using live sensor, performance or activity data

To simulate an example of receiving & plotting real-time data, a random number generator is used which generates a random number each frame. The number is then plotted on the graph.


# Testing using the Random Number Generator (Included within this project)

Just press play :) 


# How to Use
In your actual project, first create a UI -> Canvas and name it "GraphCanvas" or anything else you like. This will be the parent of all our further GameObjects. You can place it in your game scene where you want the graph to be and the size of this canvas determines the visual size of the graph. User can edit change these as per requirement.
Inside the "GraphCanvas" create a child empty GameObject called "background". Add an image component to it, and color it with whichever color you like. This will be our outer background for the graph. Ensure that the Rect Transform is stretch along both x & y-axis for this so it fills up the entire parent canvas.
Create another child empty GameObject and call it "GraphContainer". Using this "GraphContainer" as a parent, create another "background" as a child of "GraphContainer" as described previously. This means that "GraphCanvas" will have two children ("GraphContainer" & "background" ) and one grandchild called "background" from "GraphContainer".
Change the Rect Transform of "GraphContainer" to middle-center, and then move+resize it inside its parent as required to create the area where the graph shall be displayed.
Create a new C# script and copy paste the code from the Graph_Creator script to your own project
In the "GraphCanvas" object, add this C# script as a component. 
In your assets, create a 2D sprite. A circle is recommended but you can create anything you like to serve as the representation for the data points.
Drag and drop this created sprite in the Inspector -> Graph_Creator (Script) -> Circle Sprite.
To update the graph each time a data is recorded from the sensor or other peripheral, Use the following line of code in the script which obtains the sensor data.
Graph_Creator.instance.UpdateGraph(Received_Data);
where Received_Data is the data point that is to be added to the graph and Graph_Creator is the name of script.



If one is to use this script for plotting multiple subplots simultaneously then for each plot, we need to follow the same previous steps. (Changing the name from "GraphCanvas" to what the plot is going to display is recommended). A separate copy of this script should also be created for each graph and named differently. To call the script, the command from the place where the sensor data is received shall be
Script_Name.instance.UpdateGraph(Received_Data);


# How To Change Number of Data Points Displayed at Any Given Instant
The DataPoints variable tells how many data points shall be present on the graph at any given instant. change this to change the data resolution or zoom. This can be removed from being a constant and then can be changed in-game by calling a function depending on user interaction.


# Future Possibilities
We can further add multiple plots to the same graph, create additional lists for each of the plots and update each list whenever a data point is received.

# What is not included?
x-axis & y-axis labels and legends since each user will have a different plot to cater.


