using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI; 
using System.Linq;
using System;


//--------------------------------------------------------------------------------WHAT IS THIS ? -------------------------------------------------------------
//The script to create and update a real-time graph in Unity



//------------------------------------------------------------------------------- HOW TO USE ? --------------------------------------------------------------
//In Unity, first create a UI -> Canvas and name it "GraphCanvas" or anything else you like. This will be the parent of all our further GameObjects. You can place it in your game scene where you want the graph to be and the size of this canvas determines the visual size of the graph. User can edit change these as per requirement.
//Inside the "GraphCanvas" create a child empty GameObject called "background". Add an image component to it, and color it with whichever color you like. This will be our outer background for the graph. Ensure that the Rect Transform is stretch along both x & y-axis for this so it fills up the entire parent canvas.
//Create another child empty GameObject and call it "GraphContainer". Using this "GraphContainer" as a parent, create another "background" as a child of "GraphContainer" as described previously. This means that "GraphCanvas" will have two children ("GraphContainer" & "background" ) and one grandchild called "background" from "GraphContainer".
//Change the Rect Transform of "GraphContainer" to middle-center, and then move+resize it inside its parent as required to create the area where the graph shall be displayed.
//In the "GraphCanvas" object, add this script as a component.
//In your assets, create a 2D sprite. A circle is recommended but you can create anything you like to serve as the representation for the data points.
//Drag and drop this created sprite in the Inspector -> Graph_Creator (Script) -> Circle Sprite.
//To update the graph each time a data is recorded from the sensor or other peripheral, Use the following line of code in the script which obtains the sensor data.
//Graph_Creator.instance.UpdateGraph(Received_Data);
//where Received_Data is the data point that was received and Graph_Creator is the name of script.



//If one is to use this script for plotting multiple subplots simultaneously then for each plot, we need to follow the same previous steps. (Changing the name from "GraphCanvas" to what the plot is going to display is recommended). A separate copy of this script should also be created for each graph and named differently. To call the script, the command from the place where the sensor data is received shall be
//Script_Name.instance.UpdateGraph(Received_Data);


//------------------------------------------------------------------------------ HOW TO EDIT -----------------------------------------------------------------
//the DataPoints variable tells how many data points shall be present on the graph at any given instant. change this to change the data resolution or zoom. This can be removed from being a constant and then can be changed in-game by calling a function depending on user interaction.







public class Graph_Creator : MonoBehaviour
{
   
   [SerializeField] private Sprite circleSprite; 
   private RectTransform graphContainer;
   private List<GameObject> CircleAddresses = new List<GameObject>() {};
   private List <float> list = new List<float>();
   private List <GameObject> LineAddresses = new List<GameObject>() {};
   public static Graph_Creator instance;
   
   


   const int DataPoints = 150; //Change this to vary the number of data points displayed on the graph at any given instant. More points means more rendering each frame and more CPU/GPU usage.
   




   private void Awake() {
   
   graphContainer = transform.Find("GraphContainer").GetComponent<RectTransform>(); //fetches from Unity an object named "graphContainer", rename anything in Find("") as per requirement.
      
    
    for (int i=0;i<DataPoints;i++)
    {
    
    CircleAddresses.Add(CreateCircle(new Vector2(0,0)));
    list.Add(0);
    if (i<DataPoints-1)
    {
    LineAddresses.Add(CreateLine());
    }
    
    
    }
    
   
    if(instance==null) //create an instance of this graph creator
		{
		
			instance = this;
		
		}	
		else if(instance!=this)
		{
		
			Debug.Log("Instance already exists for the score publisher, not creating anymore publishers");
			Destroy(this);  //We do not want multiple instances of the same client to be communicating with the server. One instance per client is sufficient
		
		}
    
    
    ShowGraph();
      
   }
   
   
   private GameObject CreateCircle(Vector2 Position) { 
   //creates a datapoint at the vector co-ordinates specified
   
   GameObject circle = new GameObject("circle",typeof(Image));
   circle.transform.SetParent(graphContainer,false);
   circle.GetComponent<Image>().sprite = circleSprite; //placing the sprite in the assets as the circle we want for the data points
   RectTransform tf = circle.GetComponent<RectTransform>();
   tf.anchoredPosition = Position;
   tf.sizeDelta = new Vector2(5,5); //change this for circle thickness
   tf.anchorMin = new Vector2(0,0);
   tf.anchorMax = new Vector2(0,0);
   return circle;
   
   }   
   
   
   private GameObject CreateLine(){
   
   GameObject line = new GameObject("dotConnection",typeof(Image));
   line.transform.SetParent(graphContainer,false);
   line.GetComponent<Image>().color = new Color(1,1,1,0.5f);  //Change this to change the color of the line.
   return line;
   
   
   }
   
   private void CreateLineConnection(Vector2 dotPositionA, Vector2 dotPositionB,int index) {  //connects the two dots to create a graph
   
   
   Vector2 direction = (dotPositionB - dotPositionA).normalized;
   float distance = Vector2.Distance(dotPositionA,dotPositionB);
   RectTransform tf = LineAddresses[index].GetComponent<RectTransform>();
   tf.anchorMin = new Vector2(0,0);
   tf.anchorMax = new Vector2(0,0);
   tf.sizeDelta = new Vector2(distance,3f);
   tf.anchoredPosition = dotPositionA+direction*distance*0.5f;
   tf.localEulerAngles = new Vector3(0,0,(float)Math.Atan2(direction.y,direction.x)*180f/3.142f);
   
   
  } 
  
  public void UpdateGraph(float val) //the function that communicates with the outside world.
  {
  
   list.RemoveAt(0);
   list.Add(val);
   ShowGraph();
  
  }
   
   
   
   
   
   private void ShowGraph(){
   
   float graphHeight = graphContainer.sizeDelta.y;
   float yscale = 0f;
   float step = 0f;
   float xDelta = graphContainer.sizeDelta.x/DataPoints;
   if (list.Min()<0)
   {
   	yscale = list.Max()-list.Min();
   	step = list.Min();
   	
   }
   else
   {
   
   	yscale = list.Max();
   
   }
   
   
   GameObject LastCircle = null;
   for (int i=0;i<DataPoints;i++)
   {
   
   	float xPos = i*xDelta;
   	float yPos = ((list[i]-step)/yscale)*graphHeight;
   
   	RectTransform tf = CircleAddresses[i].GetComponent<RectTransform>();
   	tf.anchoredPosition = new Vector2(xPos,yPos);
   	GameObject currentCircle = CircleAddresses[i];
   	
   	if (LastCircle != null)
   	{
   	
   	CreateLineConnection(LastCircle.GetComponent<RectTransform>().anchoredPosition, currentCircle.GetComponent<RectTransform>().anchoredPosition,i-1);
   	
   	}
   	
   	LastCircle = currentCircle;
   
   
   }
   
   
   
   
   
   }
   
   
   
   
}
