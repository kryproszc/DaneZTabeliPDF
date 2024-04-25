for (int i = 0; i < n; i++)
{
  logical_value = !((exposure_longitude[woj][mies][i] > east_lon) || (exposure_longitude[woj][mies][i] < west_lon) || (exposure_latitude[woj][mies][i] < south_lat) || (exposure_latitude[woj][mies][i] > north_lat));
  if (logical_value)
  {
    // cnt++;
    lat_sub.push_back(exposure_latitude[woj][mies][i]);
    lon_sub.push_back(exposure_longitude[woj][mies][i]);
    insu_sub.push_back(exposure_insurance[woj][mies][i]);
    reas_sub.push_back(exposure_reassurance[woj][mies][i]);
    premium_sub.push_back(exposure_sum_value[woj][mies][i]);
  }
}

// Rect searchArea = {left, top, lat_center + radius, lon_center + radius};

// getQuadTree(quadtree, woj, mies).getPointsInRange(searchArea, lat_sub, lon_sub, insu_sub, reas_sub, premium_sub);