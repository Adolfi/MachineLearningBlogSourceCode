using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

var context = new MLContext();
	
// 1. Load data from .csv.
var dataPath = Path.Combine(Environment.CurrentDirectory, "housing.csv");
var dataView = context.Data
    .LoadFromTextFile<HousingData>(dataPath, hasHeader: true, separatorChar: ',');
	
// 2. Describe your data structure.
var pipeline = context.Transforms
    .Concatenate("Features", new[] { "Size", "Bedrooms" })
    .Append(context.Transforms.CopyColumns("Label", "Price"))
    .Append(context.Regression.Trainers.Sdca());

// 3. Train your model.
var model = pipeline.Fit(dataView);

// 4. Save your model to a .zip file.
context.Model.Save(model, dataView.Schema, 
    Path.Combine(Directory.GetCurrentDirectory(), "housing.zip"));


// 1. Load the model
var loadedModel = context.Model
    .Load(Path.Combine(Directory.GetCurrentDirectory(), "housing.zip"), 
    out var modelInputSchema);

// 2. Create a prediction engine
var predictionEngine = context.Model
    .CreatePredictionEngine<HousingData, HousingPricePrediction>(loadedModel);

// 3. Predict price for sample data
var prediction = predictionEngine
    .Predict(new HousingData() { Size = 750f, Bedrooms = 3f });

// 4. Output result
Console.WriteLine($"Predicted price: {prediction.Price:C}");