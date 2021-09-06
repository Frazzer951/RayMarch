#define _USE_MATH_DEFINES

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "geometry.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

const float sphere_radius = 0.25;
const Vec3f sphere_center = { 0.5, 0.5, 0 };
const Vec3f box_size      = { 0.25, 0.25, 0.25 };
const Vec3f box_center    = { 0.5, 0.5, 0 };
const int   ray_steps     = 128;

float mod( float f, float v )
{
  if( f < 0 )
  {
    f += v;
    return mod( f, v );
  }
  else if( f >= v )
  {
    f -= v;
    return mod( f, v );
  }
  return f;
}

Vec3f mod( Vec3f p, float val )
{
  float x = mod( p.x, val );
  float y = mod( p.y, val );
  float z = mod( p.z, val );

  return { x, y, z };
}

Vec3f abs( Vec3f p )
{
  return { abs( p.x ), abs( p.y ), abs( p.z ) };
}

Vec3f max( Vec3f p, float v )
{
  return { p.x > v ? p.x : v, p.y > v ? p.y : v, p.z > v ? p.z : v };
}

float maxcomp( Vec3f p )
{
  float max = p.x;
  max       = ( p.y > max ) ? p.y : max;
  max       = ( p.z > max ) ? p.z : max;
  return max;
}

float sphere_signed_distance( const Vec3f & p )
{
  return ( mod( p, 1 ) - sphere_center ).norm() - sphere_radius;
}

float box_signed_distance( const Vec3f & p )
{
  Vec3f q       = abs( p ) - box_center - box_size;
  float maxComp = maxcomp( q );
  return max( q, 0 ).norm() + ( maxComp < 0 ) ? maxComp : 0;
}

bool sphere_trace( const Vec3f & orig, const Vec3f & dir, Vec3f & pos )
{
  pos = orig;
  for( size_t i = 0; i < ray_steps; i++ )
  {
    float d = sphere_signed_distance( pos );
    if( d < 0 ) return true;
    pos = pos + dir * std::max( d * 0.1f, .01f );
  }
  return false;
}

bool box_trace( const Vec3f & orig, const Vec3f & dir, Vec3f & pos )
{
  pos = orig;
  for( size_t i = 0; i < ray_steps; i++ )
  {
    float d = box_signed_distance( pos );
    if( d < 0 ) return true;
    pos = pos + dir * std::max( d * 0.1f, .01f );
  }
  return false;
}


Vec3f distance_field_normal_sphere( const Vec3f & pos )
{
  const float eps = 0.1;
  float       d   = sphere_signed_distance( pos );
  float       nx  = sphere_signed_distance( pos + Vec3f( eps, 0, 0 ) ) - d;
  float       ny  = sphere_signed_distance( pos + Vec3f( 0, eps, 0 ) ) - d;
  float       nz  = sphere_signed_distance( pos + Vec3f( 0, 0, eps ) ) - d;
  return Vec3f( nx, ny, nz ).normalize();
}

Vec3f distance_field_normal_box( const Vec3f & pos )
{
  const float eps = 0.1;
  float       d   = box_signed_distance( pos );
  float       nx  = box_signed_distance( pos + Vec3f( eps, 0, 0 ) ) - d;
  float       ny  = box_signed_distance( pos + Vec3f( 0, eps, 0 ) ) - d;
  float       nz  = box_signed_distance( pos + Vec3f( 0, 0, eps ) ) - d;
  return Vec3f( nx, ny, nz ).normalize();
}

int main()
{
  const int          width  = 640;
  const int          height = 480;
  const float        fov    = M_PI / 3.;
  std::vector<Vec3f> framebuffer( width * height );

  std::cout << "Calculating Rays\n";

#pragma omp parallel for
  for( size_t j = 0; j < height; j++ )
  {    // actual rendering loop
    for( size_t i = 0; i < width; i++ )
    {
      float dir_x = ( i + 0.5 ) - width / 2.;
      float dir_y = -( j + 0.5 ) + height / 2.;    // this flips the image at the same time
      float dir_z = -height / ( 2. * tan( fov / 2. ) );
      Vec3f hit;
      bool  calcSphere = false;
      bool  calcBox    = true;
      if( calcSphere && sphere_trace( Vec3f( 1, 1, 3 ), Vec3f( dir_x, dir_y, dir_z ).normalize(), hit ) )
      {                                                                            // the camera is placed to (0,0,3) and it looks along the -z axis
        Vec3f light_dir            = ( Vec3f( 10, 10, 10 ) - hit ).normalize();    // one light is placed to (10,10,10)
        float light_intensity      = std::max( 0.4f, light_dir * distance_field_normal_sphere( hit ) );
        framebuffer[i + j * width] = Vec3f( 1, 1, 1 ) * light_intensity;
      }
      else if( calcBox && box_trace( Vec3f( 0.5, 0.5, 3 ), Vec3f( dir_x, dir_y, dir_z ).normalize(), hit ) )
      {
        // the camera is placed to (0,0,3) and it looks along the -z axis
        Vec3f light_dir            = ( Vec3f( 10, 10, 10 ) - hit ).normalize();    // one light is placed to (10,10,10)
        float light_intensity      = std::max( 0.4f, light_dir * distance_field_normal_box( hit ) );
        framebuffer[i + j * width] = Vec3f( 1, 1, 1 ) * light_intensity;
      }
      else
      {
        framebuffer[i + j * width] = Vec3f( 0.2, 0.7, 0.8 );    // background color
      }
    }
  }

  std::cout << "Making PPM\n";


  std::ofstream ofs( "./out.ppm", std::ios::binary );    // save the framebuffer to file
  ofs << "P6\n"
      << width << " " << height << "\n255\n";
  for( size_t i = 0; i < height * width; ++i )
  {
    for( size_t j = 0; j < 3; j++ )
    {
      ofs << (char)( std::max( 0, std::min( 255, static_cast<int>( 255 * framebuffer[i][j] ) ) ) );
    }
  }
  ofs.close();

  std::cout << "Converting to PNG\n";

  int             x, y, n;
  unsigned char * data = stbi_load( "out.ppm", &x, &y, &n, 0 );
  stbi_write_png( "out.png", x, y, n, data, x * n );
  stbi_image_free( data );

  return 0;
}